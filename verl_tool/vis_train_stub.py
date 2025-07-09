
# path = "./gen_batch.pkl"
path = "./test_gen_batch.pkl"
if __name__ == "__main__":
    import pickle
    with open(path, "rb") as f:
        gen_batch = pickle.load(f)
    print(gen_batch)
    print(gen_batch.batch.keys())
    print(gen_batch.meta_info)
    print(gen_batch.non_tensor_batch.keys())
    exit(1)