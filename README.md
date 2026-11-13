# DNN Assignment 2 - RNN, LSTM, GRU and Transformer

**Name:** Amit Kumar Mohanty **Student ID:** 2026AJ05290

## My choices
- Hidden size **h = 64** for the RNN, LSTM and GRU (the same for all three).
- Model width **d = 128** for the language model and **d = 64** for the HAR encoder. Both use 4 heads, so d is divisible by the number of heads.
- Seeds **11, 23 and 37** for every experiment with 3 seeds.
- My ID ends in 0, an even number, so my own experiment (Problem 1, task 5c) is **gradient clipping** for the plain RNN at T = 200.
  This is set by `ID_LAST_DIGIT = 0` in the first cell of `problem1.ipynb`.

## What you need
- Python 3.10 or newer, with the packages in `requirements.txt`. Only PyTorch is used, no other deep-learning library.
- Internet access the first time. The notebooks download the data themselves into a `data/` folder:
  UCI HAR from archive.ics.uci.edu and Tiny Shakespeare from the karpathy/char-rnn repository.
- No pre-trained weights are used. All file paths are relative, so the folder can be anywhere.

## How to run
1. Run `problem1/problem1.ipynb` first, then `problem2/problem2.ipynb`. Problem 2 reads the Problem 1 results from `results.csv`.
2. Each notebook runs from top to bottom with *Restart and run all*.
3. Every finished training run is added to `results.csv` straight away, and saved in a `cache/` folder.
   If the session stops, run the notebook again: finished runs are loaded from `cache/`, and only the unfinished run starts again.
   Delete `cache/` if you want to train everything again from scratch.

**In Google Colab (T4 GPU), keeping the folder structure:** choose *Runtime > Change runtime type > T4 GPU*, upload this zip,
and run these two lines in a cell:
```
!unzip -q 2026AJ05290_DNN_Assignment.zip && cd 2026AJ05290_DNN_Assignment/problem1 && jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 problem1.ipynb
!cd 2026AJ05290_DNN_Assignment/problem2 && jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 problem2.ipynb
```
You can also open one notebook on its own in Colab. Its files then go to `/content`. Problem 2 then has no Problem 1 row to
compare with, unless you upload the `results.csv` from Problem 1 first.

**Without a GPU**, the language model is slow. Set `D_LM = 64` and `LAYERS_LM = 2` in the first cell of Problem 2.
To check quickly that everything works, set `QUICK = True` (the numbers are then meaningless).

## Run time
- **On a T4: not measured**, because no GPU was available. My rough estimate from the CPU times is 1-2 hours for both notebooks.
- **What was measured:** 13.4 hours of training on one CPU thread in total (adding problem 9.9 h, language models 2.4 h,
  HAR 1.0 h). I split these runs over 8 processes on an 8-core CPU, so it took about 2.5 hours. The finished runs are in `cache/`.
- Running both notebooks again with the saved runs takes about 1 minute.

## What does not work, or differs from the brief
- The times (`sec_per_epoch`) come from a shared CPU, not a T4. Compare them only inside one table.
- The 3-token attention example (Problem 2, task 1) is my own. The hand calculation is in the notebook and in Table 5 of the report.
- With 50 epochs the LSTM does not solve the adding problem at T = 400: all three seeds stay at the baseline.
- On a GPU the results will be slightly different from these CPU results, because GPU arithmetic is not exactly the same.
