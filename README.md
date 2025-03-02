# Running Experiments with IteraPlan

## Running `plan_with_llm.py`
To execute the script, use the following command:

```bash
python3 scripts/plan_with_llm.py --floor-plan <floor-plan-id> --exp-id <exp-id> --exp-instruction "exp-instruction"
```

## Creating an Executable Python Script
To generate an executable script, run:

```bash
python3 scripts/generate_exe.py --gpt <gpt_model> --exp <exp_title>
```

## Running the Executable and Saving Videos
To run the generated executable and save videos, use:

```bash
python3 executable_plan.py --floor-plan <floor-plan-id>
```
