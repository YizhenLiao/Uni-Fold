from typing import *
from unifold.data import residue_constants as rc

import hashlib
import pathlib

import tarfile
import requests
from tqdm import tqdm
import time
import logging
import math

logger = logging.getLogger(__name__)

import os

from unifold.msa import templates, pipeline
from unifold.msa.tools import hhsearch



class MMseqsRunner:
    """object to run mmseqs api provided with the given url."""
    tqdm_bar_fmt = '{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} est.: {remaining}]'
    err_msg = (
        'MMseqs2 API is giving errors.'
        'Please confirm your input is a valid a3m string. '
        'If error persists, please try again an hour later.'
    )
    maintain_msg = (
        'MMseqs2 API is undergoing maintenance. Please try again in a few minutes.'
    )

    def __init__(
        self,
        url: str = "https://api.colabfold.com",
        retry: int = 3,
        refresh_interval: float = 1,
        timeout: Union[float, str] = "auto",
    ) -> None:
        self.url = url
        self.retry = retry
        self.t = refresh_interval
        self.timeout = timeout
    
    def get_timeout(self, estim_time: float) -> float:
        if self.timeout == "auto":
            return 5 * estim_time
        elif self.timeout > 0:
            return self.timeout
        else:
            return 864000.

    def submit(
        self,
        a3m: str,
        endpoint: str,  # ["pair", "msa"]
        mode: str,      # ["", "env"]
    ):
        res = requests.post(f'{self.url}/ticket/{endpoint}', data={'q':a3m,'mode': mode})
        try:
            out = res.json()
        except ValueError:
            logger.error(f"Server didn't reply with json: {res.text}")
            out = {"status":"ERROR"}
        return out

    def status(self, ID):
        res = requests.get(f'{self.url}/ticket/{ID}')
        try:
            out = res.json()
        except ValueError:
            logger.error(f"Server didn't reply with json: {res.text}")
            out = {"status":"ERROR"}
        return out

    def download(self, ID, path):
        res = requests.get(f'{self.url}/result/download/{ID}')
        with open(path,"wb") as out:
            out.write(res.content)

    def run_mmseqs2(
        self, 
        a3m: str,
        output_dir: str,
        endpoint: str,
        mode: str,
        verbose: bool = False,
    ) -> int:
        tic = time.perf_counter()

        os.makedirs(output_dir, exist_ok=True)
        def _path(fname):
            return os.path.join(output_dir, fname)

        tgz_path = _path(f"out_{mode}.tar.gz")
        if os.path.isfile(tgz_path):
            if verbose:
                logger.info(f"target result {tgz_path} exists.")
            return 0

        # call mmseqs2 api
        est_time = (len(a3m) // 60) * 30
        timeout = self.get_timeout(est_time)
        retry = self.retry or math.inf
        with tqdm(total=est_time, bar_format=self.tqdm_bar_fmt, disable=not verbose) as pbar:
            while retry:
                pbar.set_description("SUBMIT")
                # Resubmit job until it goes through
                out = self.submit(a3m, endpoint, mode)
                while out["status"] in ["UNKNOWN", "RATELIMIT"]:
                    time.sleep(self.t)
                    out = self.submit(a3m, endpoint, mode)
                    toc = time.perf_counter() - tic
                    pbar.update(n=int(toc-pbar.n))
                    if toc > timeout:
                        logger.error(f"MMSeqs2 API timeout after {timeout} seconds.")
                        return -1

                if out["status"] == "ERROR":
                    logger.error(self.err_msg)
                    return -1

                if out["status"] == "MAINTENANCE":
                    logger.error(self.maintain_msg)
                    return -1

                # wait for job to finish
                ID, status= out["id"], out["status"]
                pbar.set_description(status)
                while status in ["UNKNOWN","RUNNING","PENDING"]:
                    time.sleep(self.t)
                    out = self.status(ID)
                    status = out["status"]
                    pbar.set_description(status)
                    toc = time.perf_counter() - tic
                    pbar.update(n=int(toc-pbar.n))
                    if toc > timeout:
                        logger.error(f"MMSeqs2 API timeout after {timeout} seconds.")
                        return -1

                if out["status"] == "COMPLETE":
                    retry = 0

                if out["status"] == "ERROR":
                    logger.error(self.err_msg)
                    return -1
            
            pbar.set_description("DOWNLOAD")
            # Download results
            self.download(ID, tgz_path) # TODO: catch download errors.
            toc = time.perf_counter() - tic
            pbar.update(n=int(toc-pbar.n))

        if verbose:
            logger.info(f"Successfully obtained MMSeqs2 API results in {toc:.2f} seconds.")
        os.system(f"tar zxf {tgz_path} -C {output_dir}")

        return 0

    def retrieve_templates(
        self,
        template_pdb_ids: List[str],
        output_path: str,
    ):
        # TODO: catch errors.
        line = ",".join(template_pdb_ids)
        ret = os.system(f"curl -s -L {self.url}/template/{line} | tar xzf - -C {output_path}/")
        return ret

