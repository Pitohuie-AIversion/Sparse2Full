import json
import glob

files = glob.glob("/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/**/test_results.json", recursive=True)
for f in files:
    try:
        with open(f) as fp:
            data = json.load(fp)
            if 'final_test_metrics' in data:
                m = data['final_test_metrics']
                if abs(m.get('rel_l2', 0) - 0.1986) < 0.001 or abs(m.get('rel_l2', 0) - 0.2824) < 0.001:
                    print(f"FOUND Interp in {f}: {m.get('rel_l2')}")
    except:
        pass
