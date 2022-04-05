# relative improvement algorithm
def differences(numerical_items):
    # get pairwise elements
    for prev, each in zip(numerical_items[0:-1], numerical_items[1:]):
        yield each - prev

def relative_improvement_coefficients(stacked_losses):
    for each_stack in stacked_losses:
        differences = to_tensor(differences(each_stack))
        improvement = (differences.mean() + differences.median())/2
        
        