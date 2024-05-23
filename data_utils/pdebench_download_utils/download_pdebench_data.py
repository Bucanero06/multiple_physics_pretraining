import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
from torchvision.datasets.utils import download_url

def parse_metadata(pde_names):
    """
    This function parses the argument to filter the metadata of files that need to be downloaded.

    Args:
    pde_names: List containing the name of the PDE to be downloaded

    Options for pde_names:
    - Advection
    - Burgers
    - 1D_CFD
    - Diff-Sorp
    - 1D_ReacDiff
    - 2D_CFD
    - Darcy
    - 2D_ReacDiff
    - NS_Incom
    - SWE
    - 3D_CFD

    Returns:
    pde_df : Filtered dataframe containing metadata of files to be downloaded
    """
    meta_df = pd.read_csv("/data_utils/pdebench_data_urls.csv")

    # Ensure the pde_name is defined
    pde_list = [
        "advection",
        "burgers",
        "1d_cfd",
        "diff_sorp",
        "1d_reacdiff",
        "2d_cfd",
        "darcy",
        "2d_reacdiff",
        "ns_incom",
        "swe",
        "3d_cfd",
    ]

    assert all([name.lower() in pde_list for name in pde_names]), "PDE name not defined."

    # Filter the files to be downloaded
    meta_df["PDE"] = meta_df["PDE"].str.lower()
    pde_df = meta_df[meta_df["PDE"].isin(pde_names)]

    return pde_df

def download_file(root_folder, row):
    file_path = os.path.join(root_folder, row["Path"])
    download_url(row["URL"], file_path, row["Filename"], md5=row["MD5"])

def download_data(root_folder, pde_name, max_workers=os.cpu_count()):
    """
    Download data splits specific to a given PDE.

    Args:
    root_folder: The root folder where the data will be downloaded
    pde_name   : The name of the PDE for which the data to be downloaded
    max_workers: The maximum number of threads to use for downloading
    """

    print(f"Downloading data for {pde_name} ...")

    # Load and parse metadata csv file
    pde_df = parse_metadata(pde_name)

    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(download_file, root_folder, row): row for _, row in pde_df.iterrows()}

        for future in tqdm(as_completed(future_to_url), total=len(future_to_url)):
            try:
                future.result()
            except Exception as exc:
                row = future_to_url[future]
                print(f"{row['Filename']} generated an exception: {exc}")

    print("Download complete!")

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        prog="Download Script",
        description="Helper script to download the PDEBench datasets",
        epilog="",
    )

    arg_parser.add_argument(
        "--root_folder",
        type=str,
        required=True,
        help="Root folder where the data will be downloaded",
    )
    arg_parser.add_argument(
        "--pde_name",
        action="append",
        help="Name of the PDE dataset to download. You can use this flag multiple times to download multiple datasets",
    )

    args = arg_parser.parse_args()

    download_data(args.root_folder, args.pde_name)
