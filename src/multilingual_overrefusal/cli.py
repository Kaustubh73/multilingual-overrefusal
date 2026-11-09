"""Command line: `multilingual-overrefusal run configs/<name>.toml [--only STEP] [--force STEP] [--set a.b=c]`."""
import argparse

from .runner import apply_overrides, load_config, run


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="multilingual-overrefusal")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the steps listed in a TOML config")
    r.add_argument("config")
    r.add_argument("--only", help="run just this step (even if already done)")
    r.add_argument("--force", action="append", default=[], help="rerun this step (repeatable)")
    r.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value, e.g. generate.backend=modal")
    a = ap.parse_args(argv)
    cfg = apply_overrides(load_config(a.config), a.overrides)
    run(cfg, only=a.only, force=a.force)


if __name__ == "__main__":
    main()
