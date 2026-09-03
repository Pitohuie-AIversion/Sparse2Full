import json

with open("files_sw.txt") as f:
    files = [line.strip() for line in f]

for f in files:
    try:
        with open(f) as fp:
            data = json.load(fp)
            if 'final_test_metrics' in data:
                m = data['final_test_metrics']
                print(f"{f}: Rel_L2: {m.get('rel_l2'):.4f}, psnr: {m.get('psnr'):.2f}, ssim: {m.get('ssim'):.4f}, h_err: {m.get('dc_error'):.4f}")
    except:
        pass
