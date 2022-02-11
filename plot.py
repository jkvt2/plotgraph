import matplotlib.pyplot as plt

def read_csv(path):
    with open(path + '.csv', 'r') as f:
        header = f.readline().rstrip('\n')
        fields = header.split(',')
        results = {}
        for field in fields:
            results[field] = []
        for l in f:
            vals = dict(zip(fields, map(float, l.rstrip('\n').split(','))))
            for field in fields:
                results[field].append(vals[field])
    return results

def round_to(i, k=5):
    return k * ((i + k/2) // k)

x = 'epoch'
noises = {
    0: '0_0',
    2: '2_5',
    4: '4_5',
    6: '6_5'}
h2f = {
    'rn50': 'ResNet-50',
    'vits': 'ViT-S',
    'swint': 'Swin-T',
    'mixer': 'MLP-Mixer',
    'gmlp': 'gMLP-S',
}
results = {}
for shortform, filename in noises.items():
    results[shortform] = read_csv(filename)

def plot_across_nl():
    fig, axs = plt.subplots(1, 4)
    miny, maxy = 100, 0
    for i, n in enumerate(noises.keys()):
        ax = axs[i]
        for arch, label in h2f.items():
            ax.plot(results[n][x], results[n][arch], label=label)
            minref = min(results[n][arch][100], results[n][arch][-1])
            maxref = max(results[n][arch][100], results[n][arch][-1])
            if minref < miny:
                miny = minref
            if maxref > maxy:
                maxy = maxref
        ax.set_xlabel('Epoch')
        ax.set_title(f'NL={n}')
    for ax in axs:
        ax.set_ylim([round_to(miny - 2, 2), round_to(maxy + 2, 2)])
    for ax in axs[1:]:
        ax.set_yticks([])
    handles, labels = axs[0].get_legend_handles_labels()
    axs[-1].legend(handles, labels, loc=[1.1,0.3])

    fig.text(0.09, 0.6, 'Top 1 val acc (%)', va='center', rotation='vertical')
    plt.subplots_adjust(wspace=0.05, bottom=0.2)
    plt.show()

def plot_across_arch():
    fig, axs = plt.subplots(1, 3)
    miny, maxy = 100, 0
    for i, arch in enumerate(['rn50', 'vits', 'swint']):
        ax = axs[i]
        for n, fn in noises.items():
            nl, nm = fn.split('_')
            label = f'NL={nl}, NM={nm}'
            ax.plot(results[n][x], results[n][arch], label=label)
            minref = min(results[n][arch][100], results[n][arch][-1])
            maxref = max(results[n][arch][100], results[n][arch][-1])
            if minref < miny:
                miny = minref
            if maxref > maxy:
                maxy = maxref
        ax.set_xlabel('Epoch')
        ax.set_title(h2f[arch])
    for ax in axs:
        ax.set_ylim([round_to(miny - 2, 2), round_to(maxy + 2, 2)])
    for ax in axs[1:]:
        ax.set_yticks([])
    handles, labels = axs[0].get_legend_handles_labels()
    axs[-1].legend(handles, labels, loc=[1.05,0.3])

    fig.text(0.09, 0.6, 'Top 1 val acc (%)', va='center', rotation='vertical')
    plt.subplots_adjust(wspace=0.05, bottom=0.2)
    plt.show()

plot_across_nl()
plot_across_arch()
