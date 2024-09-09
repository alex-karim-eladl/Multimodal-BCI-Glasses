import logging
import json

from config import *

log = logging.getLogger(__name__)



def init_logger() -> logging.Logger:
    # if config is not None:
    #     # config = Path('configs/logger.yaml')
    #     with config.open() as file:
    #         config = yaml.safe_load(file)

    #     logging.config.dictConfig(config)
    snirf_log = logging.getLogger('pysnirf2')
    # snirf_log.setLevel(logging.CRTICAL+1)
    snirf_log.disabled = True

    logging.getLogger('close').disabled = True
    logging.getLogger('appendGroup').disabled = True

    debug_file = DATA_DIR / 'logs/debug.log'
    info_file = DATA_DIR / 'logs/info.log'
    debug_file.parent.mkdir(parents=True, exist_ok=True)

    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    log.handlers.clear()

    formatter = logging.Formatter('%(module)s %(levelname)s %(name)s.%(funcName)s:%(lineno)i - %(message)s %(asctime)s')

    fh = logging.FileHandler(debug_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)

    # fh = logging.FileHandler(info_file)
    # fh.setLevel(logging.INFO)
    # fh.setFormatter(formatter)
    # log.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    sh.addFilter(logging.Filter('pysnirf2'))
    log.addHandler(sh)

    return log

