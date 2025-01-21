# from __future__ import annotations

# from lm_eval.models.sae_steered_beta import InterventionModelLM


# def steered_factory():
#     import torch

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     return InterventionModelLM(
#         csv_path="../examples/dog_steer.csv",
#         base_name="google/gemma-2-2b",
#         device=device,
#         dtype="float32",
#     )


# import os
# import sys
# from pathlib import Path

# import numpy as np
# import tokenizers
# import torch
# from packaging.version import parse as parse_version

# from lm_eval import tasks
# from lm_eval.api.instance import Instance


# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# task_manager = tasks.TaskManager()

# TEST_STRING = "foo bar"


# class Test_SteeredLM:
#     torch.use_deterministic_algorithms(True)
#     task_list = task_manager.load_task_or_group(["arc_easy", "gsm8k", "wikitext"])
#     version_minor = sys.version_info.minor
#     multiple_choice_task = task_list["arc_easy"]  # type: ignore
#     multiple_choice_task.build_all_requests(limit=10, rank=0, world_size=1)
#     MULTIPLE_CH: list[Instance] = multiple_choice_task.instances
#     generate_until_task = task_list["gsm8k"]  # type: ignore
#     generate_until_task._config.generation_kwargs["max_gen_toks"] = 10
#     generate_until_task.set_fewshot_seed(1234)  # fewshot random generator seed
#     generate_until_task.build_all_requests(limit=10, rank=0, world_size=1)
#     generate_until: list[Instance] = generate_until_task.instances
#     rolling_task = task_list["wikitext"]  # type: ignore
#     rolling_task.build_all_requests(limit=10, rank=0, world_size=1)
#     ROLLING: list[Instance] = rolling_task.instances

#     # Update expected results for the steered model
#     # Note: These are placeholder values - you'll need to replace with actual results
#     MULTIPLE_CH_RES = [0.0] * 40  # Replace with actual results
#     generate_until_RES = [""] * 10  # Replace with actual results
#     ROLLING_RES = [0.0] * 10  # Replace with actual results

#     # Update model initialization to use InterventionModelLM
#     LM = steered_factory()

#     def test_logliklihood(self) -> None:
#         res = self.LM.loglikelihood(self.MULTIPLE_CH)
#         _RES, _res = self.MULTIPLE_CH_RES, [r[0] for r in res]
#         # log samples to CI
#         dir_path = Path("test_logs")
#         dir_path.mkdir(parents=True, exist_ok=True)

#         file_path = dir_path / f"outputs_log_{self.version_minor}.txt"
#         file_path = file_path.resolve()
#         with open(file_path, "w", encoding="utf-8") as f:
#             f.write("\n".join(str(x) for x in _res))
#         assert np.allclose(_res, _RES, atol=1e-2)
#         # check indices for Multiple Choice
#         argmax_RES, argmax_res = (
#             np.argmax(np.array(_RES).reshape(-1, 4), axis=1),
#             np.argmax(np.array(_res).reshape(-1, 4), axis=1),
#         )
#         assert (argmax_RES == argmax_res).all()

#     def test_generate_until(self) -> None:
#         res = self.LM.generate_until(self.generate_until)
#         assert res == self.generate_until_RES

#     def test_logliklihood_rolling(self) -> None:
#         res = self.LM.loglikelihood_rolling(self.ROLLING)
#         assert np.allclose(res, self.ROLLING_RES, atol=1e-1)

#     def test_toc_encode(self) -> None:
#         res = self.LM.tok_encode(TEST_STRING)
#         assert res == [12110, 2534]

#     def test_toc_decode(self) -> None:
#         res = self.LM.tok_decode([12110, 2534])
#         assert res == TEST_STRING

#     def test_batch_encode(self) -> None:
#         res = self.LM.tok_batch_encode([TEST_STRING, "bar foo"])[0].tolist()
#         assert res == [[12110, 2534], [2009, 17374]]

#     def test_model_generate(self) -> None:
#         context = self.LM.tok_batch_encode([TEST_STRING])[0]
#         res = self.LM._model_generate(context, max_length=10, stop=["\n\n"])
#         res = self.LM.tok_decode(res[0])
#         if parse_version(tokenizers.__version__) >= parse_version("0.20.0"):
#             assert res == "foo bar\n<bazhang> !info bar"
#         else:
#             assert res == "foo bar\n<bazhang>!info bar"
