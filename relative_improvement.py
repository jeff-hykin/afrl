# relative improvement algorithm
def differences(numerical_items):
    # get pairwise elements
    for prev, each in zip(numerical_items[0:-1], numerical_items[1:]):
        yield each - prev

def get_normalizing_coefficients(row_of_losses):
    row_of_losses = to_tensor(row_of_losses)
    max_value = row_of_losses.max()
    normalizing_coefficients = []
    for each in row_of_losses:
        if each == max_value:
            normalizing_coefficients.append(1)
        else:
            normalizing_coefficients.append(max_value/each)
    return normalizing_coefficients


from collections import defaultdict
actual_coefficient_at = defaultdict(default=lambda : 1)
def get_stablizing_coefficients(stacked_losses, lookback, discount_rate)
    most_recent_row = [ each_stack[-1] for each_stack in stacked_losses ]
    
    first_loss = len(stacked_losses[0]) == 1
    if first_loss:
        # initialize
        for loss_index, each_coefficient in enumerate(get_normalizing_coefficients(most_recent_row)):
            actual_coefficient_at[loss_index, 0] = each_coefficient
    
        
    for loss_index, each_stack in enumerate(stacked_losses):
        for timestep_index, each_loss_value in enumerate(each_stack):
            actual_coefficient_at[loss_index, timestep_index]
    


# When stable -> ignore it
def relative_improvement_coefficients(stacked_losses):
    most_recent_row = [ each_stack[-1] for each_stack in stacked_losses ]
    
    for each_stack in stacked_losses:
        differences = to_tensor(differences(each_stack))
        improvement = (differences.mean() + differences.median())/2
        
        