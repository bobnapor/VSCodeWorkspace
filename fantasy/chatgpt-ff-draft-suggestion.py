ppr = float(input("Enter the player's PPR value: "))
value = float(input("Enter the player's projected value: "))
teams = int(input("Enter the number of teams in the draft: "))

total_budget = teams * 200  # total budget for all teams
remaining_budget = total_budget - 1  # subtract 1 for the current team
remaining_players = (teams - 1) * 16  # number of remaining roster spots

# calculate the average cost per remaining player
if remaining_players > 0:
    avg_cost = remaining_budget / remaining_players
else:
    avg_cost = 0

# calculate the player's value above replacement
par = avg_cost * 16  # projected auction value of a replacement-level player
var = value - par  # value above replacement

# calculate the player's optimal price
optimal_price = var + par + (ppr * 0.1)  # add 10 cents for each PPR point

print(f"The optimal price for this player is ${optimal_price:.2f}")