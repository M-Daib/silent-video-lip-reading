import ssl
import urllib.request
import os

# Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

base_url = "https://spandh.dcs.shef.ac.uk/gridcorpus/examples/"
files = [
    "s2_swwp2s.wav",
    "id2_vcd_swwp2s.mpg",
    "id2_6000_swwp2s.mpg",
    "swwp2s.align",
    "s23_priazn.wav",
    "id23_vcd_priazn.mpg",
    "id23_6000_priazn.mpg",
    "priazn.align"
]

output_dir = "grid_corpus/examples"
os.makedirs(output_dir, exist_ok=True)

for file in files:
    url = base_url + file
    output_path = os.path.join(output_dir, file)
    print(f"Downloading {url} to {output_path}")
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"Successfully downloaded {file}")
    except Exception as e:
        print(f"Failed to download {file}: {e}")