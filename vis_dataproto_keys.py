# path = "./gen_batch.pkl"
path = "./test_gen_batch.pkl"
# path = "./chat_scheduler_output_batch1.pkl"
# path = "./chat_scheduler_output_batch2.pkl"
# path = "./run_llm_loop_async_batch1.pkl"
path = "./run_llm_loop_async_batch2.pkl"
if __name__ == "__main__":
    import pickle
    with open(path, "rb") as f:
        gen_batch = pickle.load(f)
    # print(gen_batch)
    print(gen_batch.batch.keys())
    print(gen_batch.meta_info)
    print(gen_batch.non_tensor_batch.keys())
    exit(1)