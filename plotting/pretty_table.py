from prettytable import PrettyTable
table = PrettyTable()

table.field_names = ["Dataset", "AR drafter", "Fast-dLLM", "FailFast (ours)"]
table.add_row(["MATH", 2.89, 3.51, 4.78])
table.add_row(["AIME", 2.80, 3.15, 4.25])

print(table)