import logging
from itertools import combinations
from math import factorial
from multiprocessing import Pool

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

data = [
{"Rank": 5, "Stars": 7, "Main_3_Score": 267},
{"Rank": 12, "Stars": 7, "Main_3_Score": 798.333333333333},
{"Rank": 18, "Stars": 6, "Main_3_Score": 789},
{"Rank": 20, "Stars": 6, "Main_3_Score": 742},
{"Rank": 21, "Stars": 7, "Main_3_Score": 777.666666666667},
{"Rank": 22, "Stars": 7, "Main_3_Score": 724},
{"Rank": 24, "Stars": 6, "Main_3_Score": 722.333333333333},
{"Rank": 24, "Stars": 7, "Main_3_Score": 708.666666666667},
{"Rank": 25, "Stars": 7, "Main_3_Score": 686},
{"Rank": 27, "Stars": 6, "Main_3_Score": 733},
{"Rank": 27, "Stars": 6, "Main_3_Score": 660.666666666667},
{"Rank": 29, "Stars": 7, "Main_3_Score": 744.333333333333},
{"Rank": 29, "Stars": 6, "Main_3_Score": 672.333333333333},
{"Rank": 30, "Stars": 5, "Main_3_Score": 646.666666666667},
{"Rank": 31, "Stars": 7, "Main_3_Score": 715.666666666667},
{"Rank": 32, "Stars": 5, "Main_3_Score": 681.666666666667},
{"Rank": 34, "Stars": 6, "Main_3_Score": 672},
{"Rank": 35, "Stars": 6, "Main_3_Score": 697},
{"Rank": 35, "Stars": 6, "Main_3_Score": 658.333333333333},
{"Rank": 36, "Stars": 3, "Main_3_Score": 602.333333333333},
{"Rank": 37, "Stars": 7, "Main_3_Score": 638.666666666667},
{"Rank": 39, "Stars": 6, "Main_3_Score": 599},
{"Rank": 40, "Stars": 6, "Main_3_Score": 632.666666666667},
{"Rank": 40, "Stars": 6, "Main_3_Score": 586.666666666667},
{"Rank": 41, "Stars": 5, "Main_3_Score": 658},
{"Rank": 43, "Stars": 5, "Main_3_Score": 685},
{"Rank": 44, "Stars": 6, "Main_3_Score": 644},
{"Rank": 46, "Stars": 5, "Main_3_Score": 612},
{"Rank": 47, "Stars": 6, "Main_3_Score": 594.666666666667},
{"Rank": 48, "Stars": 6, "Main_3_Score": 568},
{"Rank": 49, "Stars": 6, "Main_3_Score": 623.666666666667},
{"Rank": 49, "Stars": 5, "Main_3_Score": 485.333333333333},
{"Rank": 51, "Stars": 5, "Main_3_Score": 659.333333333333},
{"Rank": 52, "Stars": 7, "Main_3_Score": 599.666666666667},
{"Rank": 54, "Stars": 6, "Main_3_Score": 534.666666666667},
{"Rank": 54, "Stars": 6, "Main_3_Score": 471.333333333333},
{"Rank": 55, "Stars": 6, "Main_3_Score": 650},
{"Rank": 56, "Stars": 6, "Main_3_Score": 563},
{"Rank": 57, "Stars": 6, "Main_3_Score": 606.666666666667},
{"Rank": 58, "Stars": 4, "Main_3_Score": 510},
{"Rank": 59, "Stars": 6, "Main_3_Score": 577.333333333333},
{"Rank": 61, "Stars": 5, "Main_3_Score": 577.333333333333},
{"Rank": 63, "Stars": 5, "Main_3_Score": 696.333333333333},
{"Rank": 64, "Stars": 6, "Main_3_Score": 599},
{"Rank": 63, "Stars": 5, "Main_3_Score": 524},
{"Rank": 64, "Stars": 5, "Main_3_Score": 523.333333333333},
{"Rank": 65, "Stars": 5, "Main_3_Score": 612},
{"Rank": 66, "Stars": 6, "Main_3_Score": 557.666666666667},
{"Rank": 67, "Stars": 6, "Main_3_Score": 651.333333333333},
{"Rank": 69, "Stars": 3, "Main_3_Score": 459},
{"Rank": 70, "Stars": 4, "Main_3_Score": 488.333333333333},
{"Rank": 71, "Stars": 5, "Main_3_Score": 547.666666666667},
{"Rank": 72, "Stars": 5, "Main_3_Score": 611.666666666667},
{"Rank": 73, "Stars": 4, "Main_3_Score": 519},
{"Rank": 73, "Stars": 5, "Main_3_Score": 440},
{"Rank": 75, "Stars": 4, "Main_3_Score": 558},
{"Rank": 75, "Stars": 4, "Main_3_Score": 511.666666666667},
{"Rank": 76, "Stars": 5, "Main_3_Score": 549.666666666667},
{"Rank": 78, "Stars": 6, "Main_3_Score": 561},
{"Rank": 79, "Stars": 5, "Main_3_Score": 560.333333333333},
{"Rank": 80, "Stars": 4, "Main_3_Score": 536.333333333333},
{"Rank": 81, "Stars": 4, "Main_3_Score": 501.333333333333},
{"Rank": 83, "Stars": 5, "Main_3_Score": 575.666666666667},
{"Rank": 83, "Stars": 4, "Main_3_Score": 569.333333333333},
{"Rank": 84, "Stars": 5, "Main_3_Score": 496.666666666667},
{"Rank": 85, "Stars": 5, "Main_3_Score": 455},
{"Rank": 86, "Stars": 4, "Main_3_Score": 558},
{"Rank": 87, "Stars": 5, "Main_3_Score": 379},
{"Rank": 88, "Stars": 6, "Main_3_Score": 554},
{"Rank": 90, "Stars": 5, "Main_3_Score": 463.666666666667},
{"Rank": 90, "Stars": 4, "Main_3_Score": 492},
{"Rank": 91, "Stars": 4, "Main_3_Score": 453.333333333333},
{"Rank": 92, "Stars": 4, "Main_3_Score": 451},
{"Rank": 93, "Stars": 5, "Main_3_Score": 455.333333333333},
{"Rank": 93, "Stars": 4, "Main_3_Score": 505.333333333333},
{"Rank": 95, "Stars": 4, "Main_3_Score": 468},
{"Rank": 96, "Stars": 3, "Main_3_Score": 414},
{"Rank": 96, "Stars": 3, "Main_3_Score": 489.666666666667},
{"Rank": 97, "Stars": 5, "Main_3_Score": 435},
{"Rank": 98, "Stars": 3, "Main_3_Score": 407.666666666667},
{"Rank": 100, "Stars": 4, "Main_3_Score": 449.333333333333},
{"Rank": 102, "Stars": 4, "Main_3_Score": 271},
{"Rank": 103, "Stars": 4, "Main_3_Score": 410.666666666667},
{"Rank": 102, "Stars": 4, "Main_3_Score": 469},
{"Rank": 103, "Stars": 5, "Main_3_Score": 473.333333333333},
{"Rank": 105, "Stars": 4, "Main_3_Score": 391.333333333333},
{"Rank": 106, "Stars": 4, "Main_3_Score": 408},
{"Rank": 107, "Stars": 4, "Main_3_Score": 401.666666666667},
{"Rank": 108, "Stars": 5, "Main_3_Score": 425.666666666667},
{"Rank": 109, "Stars": 4, "Main_3_Score": 398.666666666667},
{"Rank": 110, "Stars": 5, "Main_3_Score": 538},
{"Rank": 111, "Stars": 3, "Main_3_Score": 435.333333333333},
{"Rank": 112, "Stars": 4, "Main_3_Score": 455},
{"Rank": 113, "Stars": 4, "Main_3_Score": 357.333333333333},
{"Rank": 115, "Stars": 3, "Main_3_Score": 440.333333333333},
{"Rank": 115, "Stars": 3, "Main_3_Score": 329},
{"Rank": 116, "Stars": 2, "Main_3_Score": 238.666666666667},
{"Rank": 117, "Stars": 3, "Main_3_Score": 423.333333333333},
{"Rank": 118, "Stars": 3, "Main_3_Score": 441},
{"Rank": 120, "Stars": 3, "Main_3_Score": 349},
{"Rank": 119, "Stars": 4, "Main_3_Score": 425.333333333333},
{"Rank": 121, "Stars": 3, "Main_3_Score": 344},
{"Rank": 122, "Stars": 3, "Main_3_Score": 373.666666666667},
{"Rank": 123, "Stars": 3, "Main_3_Score": 350.666666666667},
{"Rank": 124, "Stars": 3, "Main_3_Score": 351.333333333333},
{"Rank": 125, "Stars": 3, "Main_3_Score": 340.666666666667},
{"Rank": 126, "Stars": 4, "Main_3_Score": 354.666666666667},
{"Rank": 127, "Stars": 5, "Main_3_Score": 253.666666666667},
{"Rank": 129, "Stars": 2, "Main_3_Score": 278},
{"Rank": 130, "Stars": 4, "Main_3_Score": 316.333333333333},
{"Rank": 130, "Stars": 2, "Main_3_Score": 387.666666666667},
{"Rank": 133, "Stars": 3, "Main_3_Score": 336.333333333333},
{"Rank": 132, "Stars": 3, "Main_3_Score": 282},
{"Rank": 133, "Stars": 2, "Main_3_Score": 122.666666666667},
{"Rank": 134, "Stars": 3, "Main_3_Score": 276},
{"Rank": 135, "Stars": 2, "Main_3_Score": 219},
{"Rank": 136, "Stars": 3, "Main_3_Score": 267},
{"Rank": 137, "Stars": 3, "Main_3_Score": 446},
{"Rank": 138, "Stars": 3, "Main_3_Score": 319},
{"Rank": 139, "Stars": 2, "Main_3_Score": 283},
{"Rank": 140, "Stars": 3, "Main_3_Score": 275},
{"Rank": 141, "Stars": 2, "Main_3_Score": 265},
{"Rank": 142, "Stars": 2, "Main_3_Score": 289.666666666667},
{"Rank": 144, "Stars": 4, "Main_3_Score": 285},
{"Rank": 144, "Stars": 2, "Main_3_Score": 347.666666666667},
{"Rank": 145, "Stars": 3, "Main_3_Score": 266.333333333333},
{"Rank": 146, "Stars": 2, "Main_3_Score": 185.333333333333},
{"Rank": 147, "Stars": 2, "Main_3_Score": 286.666666666667},
{"Rank": 148, "Stars": 2, "Main_3_Score": 180.666666666667},
{"Rank": 149, "Stars": 2, "Main_3_Score": 251.666666666667},
{"Rank": 150, "Stars": 3, "Main_3_Score": 189.666666666667},
{"Rank": 151, "Stars": 2, "Main_3_Score": 178.333333333333},
{"Rank": 151, "Stars": 2, "Main_3_Score": 151.666666666667},
{"Rank": 152, "Stars": 2, "Main_3_Score": 191},
{"Rank": 154, "Stars": 2, "Main_3_Score": 152},
{"Rank": 155, "Stars": 2, "Main_3_Score": 110},
{"Rank": 156, "Stars": 2, "Main_3_Score": 87},
{"Rank": 158, "Stars": 2, "Main_3_Score": 94.3333333333333},
{"Rank": 158, "Stars": 2, "Main_3_Score": 136.666666666667},
{"Rank": 159, "Stars": 2, "Main_3_Score": 58.3333333333333},
{"Rank": 160, "Stars": 2, "Main_3_Score": 133.666666666667},
{"Rank": 161, "Stars": 2, "Main_3_Score": 80.3333333333333},
{"Rank": 161, "Stars": 2, "Main_3_Score": 104},
{"Rank": 162, "Stars": 2, "Main_3_Score": 53.3333333333333},
{"Rank": 164, "Stars": 2, "Main_3_Score": 22},
{"Rank": 165, "Stars": 2, "Main_3_Score": 51},
{"Rank": 166, "Stars": 2, "Main_3_Score": 16.6666666666667}
]

valid_combinations_set = [
    (2, 2, 2, 2, 2), (2, 2, 2, 2, 3), (2, 2, 2, 2, 4), (2, 2, 2, 2, 5),
    (2, 2, 2, 2, 6), (2, 2, 2, 2, 7), (2, 2, 2, 3, 3), (2, 2, 2, 3, 4),
    (2, 2, 2, 3, 5), (2, 2, 2, 3, 6), (2, 2, 2, 4, 4), (2, 2, 2, 4, 5),
    (2, 2, 3, 3, 3), (2, 2, 3, 3, 4), (2, 2, 3, 3, 5), (2, 2, 3, 4, 4),
    (2, 3, 3, 3, 3), (2, 3, 3, 3, 4), (3, 3, 3, 3, 3)
]

def check_combination(combo):
    stars_tuple = tuple(sorted(entry['Stars'] for entry in combo))
    if stars_tuple in valid_combinations_set:
        return combo

if __name__ == "__main__":
    # Define the number of processes
    num_processes = 4  # Adjust as needed

    # Step 1: Filter and sort the entries based on stars and main_3_score
    stars_data = {stars: [] for stars in range(2, 8)}
    for entry in data:
        stars_data[entry['Stars']].append(entry)
    for star_entries in stars_data.values():
        star_entries.sort(key=lambda x: x['Main_3_Score'], reverse=True)

    # Step 2: Select entries based on the conditions
    selected_entries = []
    selected_entries.extend(stars_data[2])
    selected_entries.extend(stars_data[3])
    selected_entries.extend(stars_data[4][:2])
    selected_entries.extend(stars_data[5][:1])
    selected_entries.extend(stars_data[6][:1])
    selected_entries.extend(stars_data[7][:1])

    # Total number of items
    n = len(selected_entries)

    # Number of items to choose at a time
    k = 5

    # Calculate the total number of combinations
    total_combinations = factorial(n) // (factorial(k) * factorial(n - k))

    logging.info(f"Total number of combinations = {total_combinations}")

    # Create a pool of processes
    with Pool(processes=num_processes) as pool:
        # Map the combination check function to the combinations
        results = pool.map(check_combination, combinations(selected_entries, 5))

    valid_combinations = []
    # Print valid combinations
    for combo in results:
        if combo:
            valid_combinations.append(combo)
    
    logging.info(f"Total number of valid combinations = {len(valid_combinations)}")

    # Sort valid_combinations using Timsort (default sorting algorithm in Python)
    valid_combinations.sort(key=lambda x: sum(entry['Main_3_Score'] for entry in x), reverse=True)

    # Print the first 10 entries after sorting along with their sum of Main_3_Score
    logging.info("First 10 entries after sorting:")
    for i, combo in enumerate(valid_combinations[:30], 1):
        total_score = sum(entry['Main_3_Score'] for entry in combo)
        logging.info(f"{i}: Combination: {combo}, Sum of Main_3_Score: {total_score}")