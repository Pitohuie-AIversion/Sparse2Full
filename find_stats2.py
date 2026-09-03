import json

with open("files_stats2.txt") as f:
    files = [line.strip() for line in f]

for f in files:
    try:
        with open(f) as fp:
            data = json.load(fp)
            if 'rel_l2' in data:
                print(f"{f}: Rel_L2: {data['rel_l2']['mean']:.4f}, psnr: {data['psnr']['mean']:.2f}")
    except:
        pass
