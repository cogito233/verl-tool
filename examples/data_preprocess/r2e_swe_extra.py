import pandas as pd
import os
from datasets import load_dataset, load_from_disk

system_prompt = """You are a programming agent who is provided a github issue and repository bash environment and is tasked to solve certain tasks (e.g., file localization, testcase generation, code repair and editing etc) to resolve the issue.

  We have access to the following functions:

  –– BEGIN FUNCTION #1: file_editor ––
  Description:
  Custom editing tool for viewing, creating and editing files
    •	State is persistent across command calls and discussions with the user
    •	If path is a file, view displays the result of applying cat -n. If path is a directory, view lists non-hidden files and directories up to 2 levels deep
    •	The create command cannot be used if the specified path already exists as a file
    •	If a command generates a long output, it will be truncated and marked with <response clipped>
    •	The undo_edit command will revert the last edit made to the file at path

  Notes for using the str_replace command:
    •	The old_str parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
    •	If the old_str parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in old_str to make it unique
    •	The new_str parameter should contain the edited lines that should replace the old_str

  Parameters:
    1.	command (string, required)
  Allowed values: [view, create, str_replace, insert, undo_edit]
  The command to run.
    2.	path (string, required)
  Absolute path to file or directory, e.g. /testbed/file.py or /testbed.
    3.	file_text (string, optional)
  Required for the create command. Contains the content of the file to be created.
    4.	old_str (string, optional)
  Required for the str_replace command. The exact string in path to replace.
    5.	new_str (string, optional)
    •	Optional for the str_replace command to specify the replacement string.
    •	Required for the insert command to specify the string to insert.
    6.	insert_line (integer, optional)
  Required for the insert command. The new_str will be inserted after the line number specified here.
    7.	view_range (array, optional)
    •	Optional for the view command (when path is a file).
    •	If provided, specifies the line range to view, e.g. [11, 12] shows lines 11 and 12.
    •	[start_line, -1] will show all lines from start_line to the end of file.
    8.	concise (boolean, optional)
    •	Optional for the view command.
    •	Defaults to True; displays a concise skeletal view of the file. If set to False, displays the full content in the specified view_range.

  –– END FUNCTION #1 ––

  –– BEGIN FUNCTION #2: execute_bash ––
  Description:
  Execute a bash command in the terminal.

  Behavior notes:
    •	If a command may run indefinitely (long-running), consider running it in the background and redirecting output, e.g. python3 app.py > server.log 2>&1 &.
    •	If the bash command returns exit code -1, it means the process is still running. The assistant may:
    •	Call this function again with command as an empty string ("") to retrieve additional logs.
    •	Send more input to STDIN of the running process by calling this function again with command set to the text input.
    •	Send command="ctrl+c" to interrupt the currently running process.
    •	If the command times out, it will be interrupted (SIGINT). The assistant may then retry or do further steps if needed.

  Parameters:
    1.	cmd (string, required)
  The bash command (and optional arguments) to execute.
    •	Can be empty ("") to retrieve more logs if the process is still running.
    •	Can be "ctrl+c" to interrupt the running process.

  –– END FUNCTION #2 ––

  –– BEGIN FUNCTION #3: search ––
  Description:
  Search for a term in a directory or a single file.
    •	If path is a directory (or unspecified, default is .), it recursively searches all non-hidden files and directories for the search term.
    •	If path points to a file, it runs a grep -n in that file to show line numbers matching the search term.
    •	If more than 100 files match in a directory search, results are truncated and the tool will inform you to narrow your search.
    •	If no matches are found, it will inform you as well.

  Parameters:
    1.	search_term (string, required)
  The term or string to search for in files.
    2.	path (string, optional)
  The file or directory to search in. Defaults to . if not specified.

  –– END FUNCTION #3 ––

  –– BEGIN FUNCTION #4: finish ––
  Description:
  Finish the interaction once the task is complete or if no further progress can be made.

  Behavior notes:
    •	The submit command finalizes your output.

  Parameters:
    1.	command (string, required)
  Currently allowed value: [submit]
    2.	result (string, optional)
  The result text or final message to submit. Defaults to an empty string if not provided.

  –– END FUNCTION #4 ––

  If you choose to call a function ONLY reply in the following format with NO suffix:

  <function=example_function_name>
  <parameter=example_parameter_1>value_1</parameter>
  <parameter=example_parameter_2>
  This is the value for the second parameter
  that can span
  multiple lines
  </parameter>
  </function>

  <IMPORTANT>
  Reminder:
  - Function calls MUST follow the specified format, start with <function= and end with </function>
  - Required parameters MUST be specified
  - Only call one function at a time
  - VERY IMPORTANT: Each response must include both reasoning (as natural text) and function call (in above format) to solve the task.
"""

def build_instances(data, split_name):
    instances = []
    for row_i, row in enumerate(data):
        instance = {
            "data_source": "r2e_swe",
            "prompt": [
                {"role": "system", "content": system_prompt},
            ],
            "ability": "r2e_swe",
            "reward_model": {
                "style": "rule",
                "ground_truth": row["test_patch"],
            },
            "extra_info": {
                "split": split_name,
                "index": row["instance_id"],
                "id": row["instance_id"],
                "row_i": row_i,
                "ds": row
            }
        }
        # instance["extra_info"]["ds"]["is_extra_sync"] = True
        # print(instance)
        # exit(1)
        instances.append(instance)
    return instances

def build_dataset(dataset_name, dataset_path):
    print(dataset_path)
    if os.path.isdir(dataset_path):
        dataset = load_from_disk(dataset_path)
    else:
        dataset = load_dataset(dataset_path)
    print(dataset)
    train_data = dataset["train"] if "train" in dataset else dataset
    # Select first samples for debugging - ensure we don't exceed dataset size
    max_samples = 25600
    actual_size = len(train_data)
    selected_size = min(max_samples, actual_size)
    print(f"Dataset has {actual_size} samples, selecting {selected_size} samples")
    train_data = train_data.select(range(selected_size))
    print(train_data)

    # dev_data = dataset["dev"]
    # # Select first 160 samples for debugging
    # dev_data = dev_data.select(range(160))

    # test_data = dataset["test"]
    # # Select first 160 samples for debugging
    # test_data = test_data.select(range(160))

    train_instances = build_instances(train_data, "train")
    # dev_instances = build_dataset(dev_data, "dev")
    # test_instances = build_dataset(test_data, "test")

    from datasets import Dataset
    train_dataset = Dataset.from_list(train_instances)
    # dev_dataset = Dataset.from_list(dev_instances)
    # test_dataset = Dataset.from_list(test_instances)

    import argparse
    # Create a simple args object for output directory
    class Args:
        def __init__(self):
            self.output_dir = f"./data/{dataset_name}"
    args = Args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(args.output_dir, "train.parquet"))
    # dev_dataset.to_parquet(os.path.join(args.output_dir, "dev.parquet"))
    # test_dataset.to_parquet(os.path.join(args.output_dir, "test.parquet"))

    print("Done! Train size:", len(train_dataset))


if __name__ == "__main__":
    dataset_name = "r2e_swe_extra"
    dataset_path = "/minimax-dialogue/ruobai/cogito_local/r2e-gym/data/swe_extra_subsample"
    build_dataset(dataset_name, dataset_path)