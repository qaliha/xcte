import zipfile
import os
from io import BytesIO

folder_penelitian_zip = '/Users/moer/Downloads/DataPenelitian'
target_destination = './'


def is_zip_file(filename):
    return any(filename.endswith(extension) for extension in [".zip"])


penelitian_zips = [os.path.join(folder_penelitian_zip, x) for x in os.listdir(
    folder_penelitian_zip) if is_zip_file(x)]

for zip in penelitian_zips:
    with zipfile.ZipFile(zip, "r") as zf:
        for name in zf.namelist():
            if name == 'runs.zip':
                zf_runs = BytesIO(zf.read(name))
                with zipfile.ZipFile(zf_runs) as zfr:
                    zfr.extractall(target_destination)

                break
