import json

with open('x.json', 'r') as f:
    x = json.load(f)
    f.close()

for i in range(10):
    with open(f'y_{i}.json', 'r') as f:
        y = json.load(f)
        f.close()

    ans = '\\addplot+[const plot, no marks, thick, c' + f'{i+1}' + '] coordinates {'
    for j in range(0, len(x)-4, 4):
        if x[j] > 2920:
            break
        if x[j + 4] <= 2920:
            ans += f'({x[j]}, {y[j]}) '
        else:
            ans += f'({x[j]}, {y[j]})'

    ans += '};'

    print(f'%{i}', '\n', ans)
