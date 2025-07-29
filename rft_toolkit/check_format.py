import json, pathlib, difflib, pprint, itertools as it

paths = ["data/xiancai_sync_extra_sharegpt.jsonl",
         "data/xiancai_sync_extra.jsonl"]

objs = [json.loads(pathlib.Path(p).read_text().splitlines()[0]) for p in paths]

# 直接比较
print(objs[0].keys() == objs[1].keys())  # True
print(len(objs[0]["messages"]) == len(objs[1]["messages"]))  # True

# diff 某条 message
for i, (m1, m2) in enumerate(it.zip_longest(objs[0]["messages"],
                                            objs[1]["messages"])):
    if m1 != m2:
        print("First difference at index", i)
        pprint.pprint((m1, m2))
        break
else:
    print("No differences")
