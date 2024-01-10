class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        current = self.root
        for char in word:
            if char not in current.children:
                current.children[char] = TrieNode()
            current = current.children[char]
        current.is_end_of_word = True


def traverse_trie_for_matching_words(node, pattern, current, result):
    # Recursive function to traverse the trie and find matching words
    if not pattern:
        if node.is_end_of_word:
            result.append(current)
        return

    char = pattern[0]
    child = node.children.get(char)

    if char == '.':
        # If the pattern character is '.', match any character in trie
        for next_char, next_child in node.children.items():
            traverse_trie_for_matching_words(next_child, pattern[1:], current + next_char, result)
    elif char == '*':
        # Match 0 or more characters
        traverse_trie_for_matching_words(node, pattern[1:], current, result)
        for next_char, next_child in node.children.items():
            traverse_trie_for_matching_words(next_child, pattern, current + next_char, result)
    else:
        if child is not None:
            traverse_trie_for_matching_words(child, pattern[1:], current + char, result)


def longest_common_subsequence(str1, str2):
    # Function to find the longest common subsequence between two strings
    n1, n2 = len(str1), len(str2)
    dp = [[0] * (n2 + 1) for _ in range(n1 + 1)]

    # Dynamic programming to calculate LCS lengths
    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    common_subsequence = []
    i, j = n1, n2
    while i > 0 and j > 0:
        if str1[i - 1] == str2[j - 1]:
            common_subsequence.append(str1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return ''.join(reversed(common_subsequence))


def isMatch(s: str, p: str) -> bool:
    # Function to match strings using regular expression pattern
    s, p = ' ' + s, ' ' + p
    len_s, len_p = len(s), len(p)

    dp = [[0] * len_p for _ in range(len_s)]
    dp[0][0] = 1

    for j in range(1, len_p):
        if p[j] == '*':
            dp[0][j] = dp[0][j - 2]

    for i in range(1, len_s):
        for j in range(1, len_p):
            if p[j] in {s[i], '.'}:
                dp[i][j] = dp[i - 1][j - 1]
            elif p[j] == '*':
                dp[i][j] = dp[i][j - 2] or (dp[i - 1][j] and p[j - 1] in {s[i], '.'})

    return bool(dp[-1][-1])


def main():
    with open("input.txt", "r") as file:
        n = int(file.readline())
        trie = Trie()
        dictionary = [file.readline().strip().lower() for _ in range(n)]

        regex_pattern = file.readline().strip().lower()

    matched_words = []
    if regex_pattern.__contains__("*"):
        # Use regular expression matching for patterns with '*'
        matched_words = [word for word in dictionary if isMatch(word, regex_pattern)]
        matched_words = sorted(matched_words)
    else:
        # Use Trie for pattern matching
        for word in dictionary:
            trie.insert(word)
        traverse_trie_for_matching_words(trie.root, regex_pattern, "", matched_words)

    if len(matched_words) == 0:
        return
    else:
        # Find longest common subsequence among matched words
        lcs = matched_words[0]
        for i in range(1, min(3, len(matched_words))):
            lcs = longest_common_subsequence(lcs, matched_words[i])

    with open("output.txt", "w") as file:
        file.write(lcs)


if __name__ == "__main__":
    main()
