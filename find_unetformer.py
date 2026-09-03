import json
import glob

files = glob.glob("/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/**/test_results.json", recursive=True)
for f in files:
    try:
        with open(f) as fp:
            data = json.load(fp)
            if 'final_test_metrics' in data:
                m = data['final_test_metrics']
                if abs(m.get('rel_l2', 0) - 0.9473) < 0.001:
                    print(f"FOUND UNetFormer in {f}")
    except:
        pass
