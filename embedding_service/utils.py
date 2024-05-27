from collections import Counter

def find_most_common_person_id(results):
    person_count = Counter()
    
    # Iterate through the results and count occurrences of person_id
    for result in results:
        for item in result:
            person_id = item["entity"]["person_id"]
            person_count[person_id] += 1
    
    # Find the most common person_id and its count
    most_common = person_count.most_common(1)
    most_common_person_id, count = most_common[0] if most_common else (None, 0)
    
    return most_common_person_id, count