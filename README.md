To run plan_with_llm.py
python3 scripts/plan_with_llm.py --floor-plan <floor-plan-id> --exp-id <exp-id> --exp-instruction "exp-instruction"

To create the executable Python script,
python3 scripts/generate_exe.py --gpt <gpt_model> --exp <exp_title>

To run the exe file and save videos,
python3 executable_plan.py --floor-plan <floor-plan-id>
