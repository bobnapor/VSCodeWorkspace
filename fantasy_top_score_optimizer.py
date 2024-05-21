import itertools
import concurrent.futures
import logging
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

data = [
{"Rank": 1, "Stars": 7, "Main_3_Score": 1000},
{"Rank": 2, "Stars": 7, "Main_3_Score": 971},
{"Rank": 3, "Stars": 7, "Main_3_Score": 971},
{"Rank": 4, "Stars": 7, "Main_3_Score": 951},
{"Rank": 5, "Stars": 7, "Main_3_Score": 926},
{"Rank": 6, "Stars": 7, "Main_3_Score": 863},
{"Rank": 7, "Stars": 7, "Main_3_Score": 853},
{"Rank": 8, "Stars": 7, "Main_3_Score": 893},
{"Rank": 9, "Stars": 7, "Main_3_Score": 897},
{"Rank": 10, "Stars": 7, "Main_3_Score": 760},
{"Rank": 11, "Stars": 7, "Main_3_Score": 798},
{"Rank": 12, "Stars": 7, "Main_3_Score": 782},
{"Rank": 13, "Stars": 7, "Main_3_Score": 842},
{"Rank": 14, "Stars": 7, "Main_3_Score": 793},
{"Rank": 15, "Stars": 7, "Main_3_Score": 843},
{"Rank": 16, "Stars": 7, "Main_3_Score": 799},
{"Rank": 17, "Stars": 7, "Main_3_Score": 823},
{"Rank": 18, "Stars": 7, "Main_3_Score": 646},
{"Rank": 19, "Stars": 7, "Main_3_Score": 751},
{"Rank": 20, "Stars": 7, "Main_3_Score": 700},
{"Rank": 21, "Stars": 7, "Main_3_Score": 722},
{"Rank": 22, "Stars": 7, "Main_3_Score": 517},
{"Rank": 24, "Stars": 7, "Main_3_Score": 759},
{"Rank": 24, "Stars": 6, "Main_3_Score": 719},
{"Rank": 25, "Stars": 7, "Main_3_Score": 766},
{"Rank": 26, "Stars": 7, "Main_3_Score": 803},
{"Rank": 27, "Stars": 7, "Main_3_Score": 730},
{"Rank": 28, "Stars": 6, "Main_3_Score": 655},
{"Rank": 29, "Stars": 6, "Main_3_Score": 647},
{"Rank": 30, "Stars": 6, "Main_3_Score": 657},
{"Rank": 31, "Stars": 6, "Main_3_Score": 672},
{"Rank": 32, "Stars": 6, "Main_3_Score": 667},
{"Rank": 33, "Stars": 6, "Main_3_Score": 622},
{"Rank": 34, "Stars": 6, "Main_3_Score": 635},
{"Rank": 35, "Stars": 6, "Main_3_Score": 569},
{"Rank": 36, "Stars": 6, "Main_3_Score": 502},
{"Rank": 37, "Stars": 6, "Main_3_Score": 639},
{"Rank": 38, "Stars": 6, "Main_3_Score": 690},
{"Rank": 39, "Stars": 6, "Main_3_Score": 474},
{"Rank": 40, "Stars": 6, "Main_3_Score": 647},
{"Rank": 41, "Stars": 6, "Main_3_Score": 643},
{"Rank": 42, "Stars": 6, "Main_3_Score": 505},
{"Rank": 43, "Stars": 6, "Main_3_Score": 751},
{"Rank": 44, "Stars": 6, "Main_3_Score": 692},
{"Rank": 45, "Stars": 6, "Main_3_Score": 666},
{"Rank": 46, "Stars": 6, "Main_3_Score": 688},
{"Rank": 47, "Stars": 5, "Main_3_Score": 417},
{"Rank": 47, "Stars": 6, "Main_3_Score": 432},
{"Rank": 48, "Stars": 6, "Main_3_Score": 662},
{"Rank": 50, "Stars": 5, "Main_3_Score": 660},
{"Rank": 51, "Stars": 6, "Main_3_Score": 542},
{"Rank": 52, "Stars": 6, "Main_3_Score": 661},
{"Rank": 53, "Stars": 6, "Main_3_Score": 602},
{"Rank": 54, "Stars": 5, "Main_3_Score": 594},
{"Rank": 55, "Stars": 5, "Main_3_Score": 584},
{"Rank": 56, "Stars": 5, "Main_3_Score": 618},
{"Rank": 57, "Stars": 5, "Main_3_Score": 612},
{"Rank": 58, "Stars": 5, "Main_3_Score": 490},
{"Rank": 60, "Stars": 5, "Main_3_Score": 635},
{"Rank": 60, "Stars": 6, "Main_3_Score": 592},
{"Rank": 61, "Stars": 5, "Main_3_Score": 618},
{"Rank": 62, "Stars": 5, "Main_3_Score": 466},
{"Rank": 63, "Stars": 5, "Main_3_Score": 537},
{"Rank": 64, "Stars": 5, "Main_3_Score": 563},
{"Rank": 65, "Stars": 6, "Main_3_Score": 602},
{"Rank": 66, "Stars": 5, "Main_3_Score": 528},
{"Rank": 67, "Stars": 5, "Main_3_Score": 474},
{"Rank": 68, "Stars": 5, "Main_3_Score": 563},
{"Rank": 69, "Stars": 5, "Main_3_Score": 681},
{"Rank": 70, "Stars": 5, "Main_3_Score": 581},
{"Rank": 71, "Stars": 5, "Main_3_Score": 514},
{"Rank": 71, "Stars": 5, "Main_3_Score": 643},
{"Rank": 73, "Stars": 5, "Main_3_Score": 485},
{"Rank": 74, "Stars": 6, "Main_3_Score": 524},
{"Rank": 75, "Stars": 5, "Main_3_Score": 439},
{"Rank": 76, "Stars": 5, "Main_3_Score": 367},
{"Rank": 77, "Stars": 5, "Main_3_Score": 568},
{"Rank": 78, "Stars": 5, "Main_3_Score": 599},
{"Rank": 79, "Stars": 5, "Main_3_Score": 420},
{"Rank": 80, "Stars": 5, "Main_3_Score": 495},
{"Rank": 81, "Stars": 5, "Main_3_Score": 508},
{"Rank": 82, "Stars": 4, "Main_3_Score": 596},
{"Rank": 83, "Stars": 4, "Main_3_Score": 517},
{"Rank": 84, "Stars": 4, "Main_3_Score": 574},
{"Rank": 85, "Stars": 4, "Main_3_Score": 382},
{"Rank": 86, "Stars": 5, "Main_3_Score": 481},
{"Rank": 87, "Stars": 4, "Main_3_Score": 530},
{"Rank": 88, "Stars": 4, "Main_3_Score": 504},
{"Rank": 89, "Stars": 4, "Main_3_Score": 549},
{"Rank": 90, "Stars": 4, "Main_3_Score": 384},
{"Rank": 91, "Stars": 4, "Main_3_Score": 476},
{"Rank": 93, "Stars": 4, "Main_3_Score": 570},
{"Rank": 93, "Stars": 3, "Main_3_Score": 409},
{"Rank": 94, "Stars": 4, "Main_3_Score": 422},
{"Rank": 95, "Stars": 4, "Main_3_Score": 464},
{"Rank": 96, "Stars": 4, "Main_3_Score": 524},
{"Rank": 97, "Stars": 4, "Main_3_Score": 409},
{"Rank": 98, "Stars": 4, "Main_3_Score": 597},
{"Rank": 99, "Stars": 4, "Main_3_Score": 485},
{"Rank": 100, "Stars": 4, "Main_3_Score": 366},
{"Rank": 101, "Stars": 4, "Main_3_Score": 409},
{"Rank": 102, "Stars": 4, "Main_3_Score": 465},
{"Rank": 103, "Stars": 4, "Main_3_Score": 400},
{"Rank": 104, "Stars": 4, "Main_3_Score": 454},
{"Rank": 105, "Stars": 4, "Main_3_Score": 493},
{"Rank": 106, "Stars": 4, "Main_3_Score": 293},
{"Rank": 107, "Stars": 4, "Main_3_Score": 402},
{"Rank": 108, "Stars": 4, "Main_3_Score": 339},
{"Rank": 109, "Stars": 4, "Main_3_Score": 414},
{"Rank": 110, "Stars": 3, "Main_3_Score": 309},
{"Rank": 111, "Stars": 4, "Main_3_Score": 494},
{"Rank": 112, "Stars": 4, "Main_3_Score": 384},
{"Rank": 113, "Stars": 3, "Main_3_Score": 421},
{"Rank": 114, "Stars": 4, "Main_3_Score": 375},
{"Rank": 115, "Stars": 3, "Main_3_Score": 423},
{"Rank": 116, "Stars": 3, "Main_3_Score": 434},
{"Rank": 117, "Stars": 3, "Main_3_Score": 280},
{"Rank": 118, "Stars": 3, "Main_3_Score": 404},
{"Rank": 119, "Stars": 3, "Main_3_Score": 287},
{"Rank": 120, "Stars": 3, "Main_3_Score": 342},
{"Rank": 121, "Stars": 3, "Main_3_Score": 291},
{"Rank": 122, "Stars": 3, "Main_3_Score": 279},
{"Rank": 123, "Stars": 3, "Main_3_Score": 341},
{"Rank": 124, "Stars": 3, "Main_3_Score": 379},
{"Rank": 125, "Stars": 3, "Main_3_Score": 418},
{"Rank": 126, "Stars": 3, "Main_3_Score": 298},
{"Rank": 127, "Stars": 3, "Main_3_Score": 251},
{"Rank": 128, "Stars": 3, "Main_3_Score": 213},
{"Rank": 129, "Stars": 3, "Main_3_Score": 250},
{"Rank": 130, "Stars": 3, "Main_3_Score": 117},
{"Rank": 131, "Stars": 3, "Main_3_Score": 360},
{"Rank": 132, "Stars": 3, "Main_3_Score": 204},
{"Rank": 133, "Stars": 3, "Main_3_Score": 232},
{"Rank": 133, "Stars": 3, "Main_3_Score": 267},
{"Rank": 135, "Stars": 3, "Main_3_Score": 311},
{"Rank": 136, "Stars": 2, "Main_3_Score": 148},
{"Rank": 137, "Stars": 3, "Main_3_Score": 325},
{"Rank": 138, "Stars": 3, "Main_3_Score": 249},
{"Rank": 139, "Stars": 2, "Main_3_Score": 367},
{"Rank": 140, "Stars": 2, "Main_3_Score": 326},
{"Rank": 141, "Stars": 2, "Main_3_Score": 334},
{"Rank": 142, "Stars": 2, "Main_3_Score": 285},
{"Rank": 143, "Stars": 2, "Main_3_Score": 300},
{"Rank": 144, "Stars": 2, "Main_3_Score": 281},
{"Rank": 145, "Stars": 2, "Main_3_Score": 143},
{"Rank": 146, "Stars": 2, "Main_3_Score": 154},
{"Rank": 147, "Stars": 2, "Main_3_Score": 275},
{"Rank": 148, "Stars": 2, "Main_3_Score": 237},
{"Rank": 149, "Stars": 2, "Main_3_Score": 271},
{"Rank": 150, "Stars": 2, "Main_3_Score": 335},
{"Rank": 151, "Stars": 2, "Main_3_Score": 289},
{"Rank": 152, "Stars": 2, "Main_3_Score": 192},
{"Rank": 153, "Stars": 2, "Main_3_Score": 176},
{"Rank": 154, "Stars": 2, "Main_3_Score": 106},
{"Rank": 155, "Stars": 2, "Main_3_Score": 111},
{"Rank": 156, "Stars": 2, "Main_3_Score": 138},
{"Rank": 157, "Stars": 2, "Main_3_Score": 49},
{"Rank": 158, "Stars": 2, "Main_3_Score": 111},
{"Rank": 159, "Stars": 2, "Main_3_Score": 92},
{"Rank": 160, "Stars": 2, "Main_3_Score": 87},
{"Rank": 161, "Stars": 2, "Main_3_Score": 64},
{"Rank": 162, "Stars": 2, "Main_3_Score": 67},
{"Rank": 163, "Stars": 2, "Main_3_Score": 31},
{"Rank": 164, "Stars": 2, "Main_3_Score": 2},
{"Rank": 165, "Stars": 2, "Main_3_Score": 0},
{"Rank": 166, "Stars": 2, "Main_3_Score": 15}
]

def process_combination(combination):
    #logging.info(f"Processing combination {combination} on thread {threading.current_thread().ident}")
    total_stars = sum(row["Stars"] for row in combination)
    if total_stars <= 15:  # Filter combinations with stars count > 15
        total_main_3_score = sum(row["Main_3_Score"] for row in combination)
        return total_main_3_score, combination
    return 0, None  # Return 0 score for invalid combinations

max_main_3_score = 0
best_combination = None

def batch_combinations(combinations, batch_size):
    iterator = iter(combinations)
    for first in iterator:
        yield [first] + list(itertools.islice(iterator, batch_size - 1))

thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=10)  # Limit the number of workers

batch_size = 10000  # Process combinations in batches of 1000
combinations = itertools.combinations(data, 5)

with thread_pool as executor:
    logging.info("Beginning futures submissions")
    batch_num = 1
    batches = batch_combinations(combinations, batch_size)
    for batch in batches:
        futures = [executor.submit(process_combination, combination) for combination in batch]
        for idx, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            total_main_3_score, combination = future.result()
            if total_main_3_score > max_main_3_score:
                max_main_3_score = total_main_3_score
                best_combination = combination
            if batch_size == idx:
                logging.info(f"Processed combination {idx} in batch {batch_num}")
        batch_num += 1

print("Best combination:", best_combination)
print("Max Main_3_Score:", max_main_3_score)