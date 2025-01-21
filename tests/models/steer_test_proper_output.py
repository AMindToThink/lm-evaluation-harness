from lm_eval import tasks
from lm_eval.models.sae_steered_beta import InterventionModelLM


def get_expected_results():
    # Initialize model and tasks
    model = InterventionModelLM(
        csv_path="../../examples/dog_steer.csv",
        base_name="google/gemma-2-2b",
        device="cuda:0",
        dtype="float32",
    )

    task_manager = tasks.TaskManager()
    task_list = task_manager.load_task_or_group(["arc_easy", "gsm8k", "wikitext"])

    # Setup multiple choice task
    multiple_choice_task = task_list["arc_easy"]
    multiple_choice_task.build_all_requests(limit=10, rank=0, world_size=1)
    multiple_ch_instances = multiple_choice_task.instances

    # Setup generate until task
    generate_until_task = task_list["gsm8k"]
    generate_until_task._config.generation_kwargs["max_gen_toks"] = 10
    generate_until_task.set_fewshot_seed(1234)
    generate_until_task.build_all_requests(limit=10, rank=0, world_size=1)
    generate_until_instances = generate_until_task.instances

    # Setup rolling task
    rolling_task = task_list["wikitext"]
    rolling_task.build_all_requests(limit=10, rank=0, world_size=1)
    rolling_instances = rolling_task.instances

    # Get results
    multiple_ch_res = [r[0] for r in model.loglikelihood(multiple_ch_instances)]
    generate_until_res = model.generate_until(generate_until_instances)
    rolling_res = model.loglikelihood_rolling(rolling_instances)

    print("\nMULTIPLE_CH_RES =", multiple_ch_res)
    print("\ngenerate_until_RES =", generate_until_res)
    print("\nROLLING_RES =", rolling_res.tolist())


if __name__ == "__main__":
    get_expected_results()
