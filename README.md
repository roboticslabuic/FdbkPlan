To run plan_with_llm.py
root@7419b7b19ff3:/app/llm_hri# python3 scripts/plan_with_llm.py --floor-plan 7 --exp-id 03 --exp-instruction "Can you make sunny side up eggs?"

To create the executable Python script,
root@7419b7b19ff3:/app/llm_hri# python3 scripts/generate_exe.py --gpt gpt_4o --exp 01_Can_you_make_me_an_omlette?_plans_10-16-2024-00-29-58

To run the exe file and save videos,
root@7419b7b19ff3:/app/llm_hri/logs/gpt_4o/01_Can_you_make_me_an_omlette?_plans_10-16-2024-00-29-58# python3 executable_plan.py --floor-plan 15
