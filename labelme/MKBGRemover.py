import argparse
import codecs
import logging
import os
import os.path as osp
import sys
import numpy as np

import yaml
from qtpy import QtCore
from qtpy import QtWidgets

__appname__ = "MKBGRemover"

__version__ = "5.6.0a0"
from app import MainWindow
from label_file import load_save_config
from config import get_config
from logger import logger
from utils import newIcon


def main():
    with open("config.yaml") as file:
        data = yaml.load(stream=file, Loader=yaml.Loader)
    dl_refine_mask = data["DEEPLEARNING_REFINE_MASK"]
    USE_MASK_REFINE = dl_refine_mask["USE_MASK_REFINE"]
    MODEL_PATH = dl_refine_mask["MODEL_PATH"]
    POST_PROCESS = dl_refine_mask["POST_PROCESS"]
    GRAY_SCALE = data["GRAY_SCALE"]
    USE_BKC_REPLACE = data["USE_BKC_REPLACE"]
    BACKGROUND_COLOR_RGB = data["BACKGROUND_COLOR_RGB"]
    DATA_OUT = data["DATA_OUT"]
    if type(USE_MASK_REFINE) != bool:
        print("USE_MASK_REFINE must be bool (True or False)")
        return
    if not os.path.isfile(MODEL_PATH):
        print("MODEL_PATH is not found")
        return
    if type(POST_PROCESS) != bool:
        print("POST_PROCESS must be bool (True of False)")
        return
    if type(GRAY_SCALE) != bool:
        print("GRAY_SCALE must be bool (True or False)")
        return
    if type(USE_BKC_REPLACE) != bool:
        print("USE_BKC_REPLACE must be bool (True or False)")
        return
    if type(BACKGROUND_COLOR_RGB) != list:
        print("BACGROUND_COLOR_RGB must be list. Example: [255, 255, 255]")
        return
    else:
        color = np.array(BACKGROUND_COLOR_RGB)
        if color.shape != (3,):
            print("BACKGROUND_COLOR_RGB must be list has 3 elements for 3 channel RGB. Example: [255, 255, 255]")
            return
    if not os.path.isdir(DATA_OUT):
        print("DATA_OUT is not found")
        return
    MainWindow.REFINE_MASK_AI = USE_MASK_REFINE
    MainWindow.GRAY_SCALE = GRAY_SCALE
    MainWindow.CONTENT_CONFIG = data
    load_save_config(use_mask_refine=USE_MASK_REFINE, model_path=MODEL_PATH, post_process=POST_PROCESS, gray_scale=GRAY_SCALE, use_bkc_replace=USE_BKC_REPLACE, background_color_rgb=BACKGROUND_COLOR_RGB, data_out=DATA_OUT)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", "-V", action="store_true", help="show version")
    parser.add_argument("--reset-config", action="store_true", help="reset qt config")
    parser.add_argument(
        "--logger-level",
        default="debug",
        choices=["debug", "info", "warning", "fatal", "error"],
        help="logger level",
    )
    parser.add_argument("filename", nargs="?", help="image or label filename")
    parser.add_argument(
        "--output",
        "-O",
        "-o",
        help="output file or directory (if it ends with .json it is "
        "recognized as file, else as directory)",
    )
    default_config_file = os.path.join(os.path.expanduser("~"), ".labelmerc")
    parser.add_argument(
        "--config",
        dest="config",
        help="config file or yaml-format string (default: {})".format(
            default_config_file
        ),
        default=default_config_file,
    )
    # config for the gui
    parser.add_argument(
        "--nodata",
        dest="store_data",
        action="store_false",
        help="stop storing image data to JSON file",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--autosave",
        dest="auto_save",
        action="store_true",
        help="auto save",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--nosortlabels",
        dest="sort_labels",
        action="store_false",
        help="stop sorting labels",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--flags",
        help="comma separated list of flags OR file containing flags",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--labelflags",
        dest="label_flags",
        help=r"yaml string of label specific flags OR file containing json "
        r"string of label specific flags (ex. {person-\d+: [male, tall], "
        r"dog-\d+: [black, brown, white], .*: [occluded]})",  # NOQA
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--labels",
        help="comma separated list of labels OR file containing labels",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--validatelabel",
        dest="validate_label",
        choices=["exact"],
        help="label validation types",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--keep-prev",
        action="store_true",
        help="keep annotation of previous frame",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        help="epsilon to find nearest vertex on canvas",
        default=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if args.version:
        print("{0} {1}".format(__appname__, __version__))
        sys.exit(0)

    logger.setLevel(getattr(logging, args.logger_level.upper()))

    if hasattr(args, "flags"):
        if os.path.isfile(args.flags):
            with codecs.open(args.flags, "r", encoding="utf-8") as f:
                args.flags = [line.strip() for line in f if line.strip()]
        else:
            args.flags = [line for line in args.flags.split(",") if line]

    if hasattr(args, "labels"):
        if os.path.isfile(args.labels):
            with codecs.open(args.labels, "r", encoding="utf-8") as f:
                args.labels = [line.strip() for line in f if line.strip()]
        else:
            args.labels = [line for line in args.labels.split(",") if line]

    if hasattr(args, "label_flags"):
        if os.path.isfile(args.label_flags):
            with codecs.open(args.label_flags, "r", encoding="utf-8") as f:
                args.label_flags = yaml.safe_load(f)
        else:
            args.label_flags = yaml.safe_load(args.label_flags)

    config_from_args = args.__dict__
    config_from_args.pop("version")
    reset_config = config_from_args.pop("reset_config")
    filename = config_from_args.pop("filename")
    output = config_from_args.pop("output")
    config_file_or_yaml = config_from_args.pop("config")
    config = get_config(config_file_or_yaml, config_from_args)

    if not config["labels"] and config["validate_label"]:
        logger.error(
            "--labels must be specified with --validatelabel or "
            "validate_label: true in the config file "
            "(ex. ~/.labelmerc)."
        )
        sys.exit(1)

    output_file = None
    output_dir = None
    if output is not None:
        if output.endswith(".json"):
            output_file = output
        else:
            output_dir = output

    translator = QtCore.QTranslator()
    translator.load(
        QtCore.QLocale.system().name(),
        osp.dirname(osp.abspath(__file__)) + "/translate",
    )
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(__appname__)
    app.setWindowIcon(newIcon("icon"))
    app.installTranslator(translator)
    win = MainWindow(
        config=config,
        filename=filename,
        output_file=output_file,
        output_dir=output_dir,
    )

    if reset_config:
        logger.info("Resetting Qt config: %s" % win.settings.fileName())
        win.settings.clear()
        sys.exit(0)

    win.showMaximized()
    win.raise_()
    sys.exit(app.exec_())


# this main block is required to generate executable by pyinstaller
if __name__ == "__main__":
    main()
