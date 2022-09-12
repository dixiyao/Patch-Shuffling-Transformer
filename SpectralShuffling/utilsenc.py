import torch

def shufflerow(tensor, axis):
    row_perm = torch.rand(tensor.shape[:axis+1]).argsort(axis).to(tensor.device)  # get permutation indices
    for _ in range(tensor.ndim-axis-1): row_perm.unsqueeze_(-1)
    row_perm = row_perm.repeat(*[1 for _ in range(axis+1)], *(tensor.shape[axis+1:]))  # reformat this for the gather operation
    return tensor.gather(axis, row_perm)

def BatchPatchPartialShuffle(x,k=0.1):
    row_perm = torch.rand((x.shape[0], x.shape[1])).argsort(1).to(x.device)
    percent = int(row_perm.shape[1] * k)
    for _ in range(x.ndim - 2): row_perm.unsqueeze_(-1)
    row_perm = row_perm.repeat(*[1 for _ in range(2)], *(x.shape[2:]))  # reformat this for the gather operation
    shuffle_part = x.gather(1, row_perm)
    keep_shffule_part = shuffle_part[:, :percent, :]

    random_part = shuffle_part[:, percent:, :]
    b, n, d = random_part.shape
    random_part = random_part.reshape(b * n, d)
    random_part = random_part[torch.randperm(random_part.shape[0]), :]
    random_part = random_part.reshape(b, n, d)

    random_part = torch.cat((keep_shffule_part,random_part),dim=1)
    random_part = PatchShuffle(random_part)

    input = random_part
    perm_back = row_perm.argsort(1)
    x = input.gather(1, perm_back)
    return x

def PatchShuffle(x):
    for bs in range(x.shape[0]):
        # random permutation
        x[bs] = x[bs][torch.randperm(x.shape[1]),:]
    return x



