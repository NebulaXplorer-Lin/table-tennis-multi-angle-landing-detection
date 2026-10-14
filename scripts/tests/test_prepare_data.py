"""Offline reconstruction, identity and archive-safety checks."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts import prepare_data as prepare


class PrepareDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.source = self.root / "source"
        payloads = {
            "pp_train_data": {"annotations/00/001.json": b'{"frames": []}',
                              "rallies_videos/00/001.mp4": b"\0\0\0\x18ftypisomvideo"},
            "pp_public_data": {"manifest.json": b'{"name":"pingpang_landing_v1_public"}',
                               "gt_public.jsonl": b'{"video_id":"v1"}\n',
                               "videos/0度/video.mp4": b"\0\0\0\x18ftypisompublic"},
        }
        manifest = {"schema": 1, "datasets": {}}
        for name, files in payloads.items():
            records = []
            for relative, content in files.items():
                path = self.source / name / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
                records.append(dict(path=relative, size=len(content), sha256=prepare.sha256(path)))
            manifest["datasets"][name] = records
        records = []
        self.games = {"training": ["game_1"], "test": ["test_1"]}
        for split, games in self.games.items():
            for game in games:
                for kind, suffix in [("videos", ".mp4"), ("annotations", ".zip")]:
                    path = self.source / "ttnet" / split / kind / (game + suffix)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if suffix == ".zip":
                        with zipfile.ZipFile(path, "w") as archive:
                            archive.writestr("ball_markup.json", json.dumps({"game": game}))
                    else:
                        path.write_bytes(b"\0\0\0\x18ftypisom" + game.encode())
                    records.append(dict(path=path.relative_to(self.source / "ttnet").as_posix(),
                                        size=path.stat().st_size, sha256=prepare.sha256(path)))
        manifest["datasets"]["ttnet"] = records
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps(manifest))
        for name, value in [("ROOT", self.root), ("DATA", self.data), ("MANIFEST", self.manifest)]:
            patcher = patch.object(prepare, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(prepare.download_ttnet, "GAMES", self.games)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def zip_pp(self, name, filename=None):
        archive_path = self.root / (filename or f"{name}.zip")
        with zipfile.ZipFile(archive_path, "w") as archive:
            for entry in prepare.entries(name):
                archive.write(self.source / name / entry["path"], f"outer/{name}/{entry['path']}")
        return archive_path

    def local_download(self, url, target, attempts, validator=prepare.download_ttnet.validate_file):
        filename = url.rsplit("/", 1)[-1]
        source = next((self.source / "ttnet").rglob(filename))
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        validator(target)

    def test_fresh_rebuild_rerun_export_and_corruption(self):
        self.zip_pp("pp_train_data", "pp_tarin_data.zip")
        self.zip_pp("pp_public_data")
        with patch.object(prepare.download_ttnet, "download", side_effect=self.local_download):
            self.assertEqual(prepare.main([]), 0)
            before = (self.data / "pp_train_data/rallies_videos/00/001.mp4").stat().st_mtime_ns
            self.assertEqual(prepare.main([]), 0)
            self.assertEqual((self.data / "pp_train_data/rallies_videos/00/001.mp4").stat().st_mtime_ns, before)
        self.assertEqual(prepare.main(["--check-only"]), 0)
        self.assertNotEqual((self.data / "ttnet/training/annotations/game_1/ball_markup.json").read_bytes(),
                            (self.data / "ttnet/test/annotations/test_1/ball_markup.json").read_bytes())
        export = self.root / "export"
        self.assertEqual(prepare.main(["--export-pp", str(export)]), 0)
        self.assertTrue((export / "pp_train_data.zip").is_file())
        (self.data / "pp_public_data/gt_public.jsonl").write_bytes(b"wrong data")
        with self.assertRaisesRegex(ValueError, "不一致"):
            prepare.main(["--check-only", "--datasets", "pp_public_data"])

    def test_legacy_directories_move_without_alias_links(self):
        self.data.mkdir()
        (self.root / "pingpang").mkdir()
        shutil.copytree(self.source / "pp_public_data", self.data / "public_data")
        self.assertEqual(prepare.main(["--datasets", "pp_public_data"]), 0)
        self.assertFalse((self.data / "public_data").exists())
        self.assertFalse((self.data / "public_data").is_symlink())
        self.assertFalse(any(path.is_symlink() for path in self.data.rglob("*")))
        self.assertEqual(prepare.main(["--check-only", "--datasets", "pp_public_data"]), 0)
        self.assertEqual((self.root / "pingpang/public_data").resolve(), self.data / "pp_public_data")

    def test_http_archive_uses_archive_validator_and_download_cache(self):
        archive = self.zip_pp("pp_train_data")

        def fetch(url, target, attempts, validator):
            self.assertEqual(url, "https://example.test/train.zip?token=test")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(archive, target)
            validator(target)

        with patch.object(prepare.download_ttnet, "download", side_effect=fetch) as download:
            prepare.main(["--datasets", "pp_train_data", "--train-source",
                          "https://example.test/train.zip?token=test"])
        download.assert_called_once()
        self.assertEqual(prepare.main(["--check-only", "--datasets", "pp_train_data"]), 0)

    def test_dry_run_and_missing_source_do_not_write(self):
        self.zip_pp("pp_train_data")
        self.zip_pp("pp_public_data")
        self.assertEqual(prepare.main(["--dry-run"]), 0)
        self.assertFalse(self.data.exists())
        with self.assertRaisesRegex(ValueError, "来源不存在"):
            prepare.main(["--datasets", "pp_train_data", "--train-source", str(self.root / "missing.zip")])

    def test_tar_and_participant_zip_sources(self):
        archive = self.root / "pp_train_data.tar.gz"
        with tarfile.open(archive, "w:gz") as stream:
            stream.add(self.source / "pp_train_data", arcname="bundle/pp_train_data")
        self.zip_pp("pp_public_data", "participant.zip")
        self.assertEqual(prepare.main(["--datasets", "pp_train_data", "pp_public_data"]), 0)
        self.assertEqual(prepare.main(["--check-only", "--datasets", "pp_train_data", "pp_public_data"]), 0)

    def test_invalid_source_does_not_install_files(self):
        (self.source / "pp_train_data/annotations/00/001.json").write_text("changed")
        with self.assertRaisesRegex(ValueError, "不一致"):
            prepare.main(["--datasets", "pp_train_data", "--train-source", str(self.source / "pp_train_data")])
        self.assertFalse((self.data / "pp_train_data").exists())

    def test_archive_traversal_and_links_rejected_before_writing(self):
        for kind in ("zip", "tar"):
            for attack in ("../escape", "/absolute", "link"):
                path = self.root / f"attack.{kind}"
                if kind == "zip":
                    with zipfile.ZipFile(path, "w") as archive:
                        archive.writestr("good", b"valid")
                        member = zipfile.ZipInfo(attack)
                        if attack == "link":
                            member.external_attr = 0o120777 << 16
                        archive.writestr(member, b"../escape")
                else:
                    with tarfile.open(path, "w") as archive:
                        member = tarfile.TarInfo("good")
                        member.size = 5
                        archive.addfile(member, io.BytesIO(b"valid"))
                        member = tarfile.TarInfo(attack)
                        if attack == "link":
                            member.type = tarfile.SYMTYPE
                            member.linkname = "../escape"
                        archive.addfile(member)
                target = self.root / "staging"
                target.mkdir(exist_ok=True)
                with self.assertRaises(ValueError):
                    prepare.extract_archive(path, target)
                self.assertFalse((target / "good").exists())


if __name__ == "__main__":
    unittest.main()
