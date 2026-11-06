def get_stable_prediction(history, threshold):
    if not history:
        return None
    counts = {}
    for letter in history:
        if letter in counts:
            counts[letter] += 1
        else:
            counts[letter] = 1

    most_common = max(counts, key = counts.get)
    if counts[most_common]/len(history) >= threshold:
        return most_common
    return None

print(get_stable_prediction([], 0.4))


