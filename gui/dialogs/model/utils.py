from tkinter import messagebox


def display_eval_metrics(eval_metrics: dict):
    msg = ''
    for metric_name, score in eval_metrics.items():
        if isinstance(score, dict):
            msg += f'{metric_name}: H: {score["H"]}%, D: {score["D"]}%, A: {score["A"]}%\n'
        else:
            msg += f'{metric_name}: {score}%\n'
    messagebox.showinfo('Training Results', msg)
