import os

filepath = 'thesis_paper/manuscript_5_chapter/chapter4_results_verification.md'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Target Table section
old_table = """| 112       | 76.56        | 0.4668          | -               | 1.0000$^{\dagger}$     |
| 96        | 56.25        | 0.6364          | -               | 1.0000$^{\dagger}$     |
| 80        | 39.06        | 0.7696          | -               | 1.0000$^{\dagger}$     |
| **64**    | **25.00**    | 0.8476          | -               | 1.0000$^{\dagger}$     |
| **48**    | **14.06**    | 0.9037          | 0.9026          | 1.0000$^{\dagger}$     |
| **32**    | **6.25**     | 0.9463          | 0.9495          | 1.0000$^{\dagger}$     |
| 16        | 1.56         | 0.9840          | 0.9820          | 1.0000$^{\dagger}$     |
| 8         | 0.39         | 1.0164          | 0.9906          | 1.0000$^{\dagger}$     |
| 4         | 0.10         | 1.0096          | 0.9924          | 1.0000$^{\dagger}$     |
| 1         | 0.01         | 1.0055          | 0.9952          | 1.0000$^{\dagger}$     |"""

new_table = """| 112       | 76.56        | 0.4668          | -               | 1.0000$^{\dagger}$     |
| 96        | 56.25        | 0.6364          | -               | 1.0000$^{\dagger}$     |
| 80        | 39.06        | 0.7696          | -               | 1.0000$^{\dagger}$     |
| **64**    | **25.00**    | 0.8476          | -               | 1.0000$^{\dagger}$     |
| **48**    | **14.06**    | 0.9037          | 0.9026          | 1.0000$^{\dagger}$     |
| **32**    | **6.25**     | 0.9463          | 0.9495          | 1.0000$^{\dagger}$     |
| 16        | 1.56         | 0.9840          | 0.9820          | 1.0000$^{\dagger}$     |
| 8         | 0.39         | 1.0164          | 0.9906          | 1.0000$^{\dagger}$     |
| 4         | 0.10         | 1.0096          | 0.9924          | 1.0000$^{\dagger}$     |
| 1         | 0.01         | 1.0055          | 0.9952          | 1.0000$^{\dagger}$     |"""

if old_table in content:
    print("Old table found! It has already been partially updated.")
else:
    print("Old table not found! Wait, let me just replace the '-' with 'OOM' or something.")

# Just replace the specific rows
content = content.replace("| 112       | 76.56        | 0.4668          | -               | 1.0000$^{\dagger}$     |", "| 112       | 76.56        | 0.4668          | OOM             | 1.0000$^{\dagger}$     |")
content = content.replace("| 96        | 56.25        | 0.6364          | -               | -                      |", "| 96        | 56.25        | 0.6364          | OOM             | 1.0000$^{\dagger}$     |")
content = content.replace("| 96        | 56.25        | 0.6364          | -               | 1.0000$^{\dagger}$     |", "| 96        | 56.25        | 0.6364          | OOM             | 1.0000$^{\dagger}$     |")
content = content.replace("| 80        | 39.06        | 0.7696          | -               | 1.0000$^{\dagger}$     |", "| 80        | 39.06        | 0.7696          | OOM             | 1.0000$^{\dagger}$     |")
content = content.replace("| **64**    | **25.00**    | 0.8476          | -               | 1.0000$^{\dagger}$     |", "| **64**    | **25.00**    | 0.8476          | OOM             | 1.0000$^{\dagger}$     |")

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
print("Table updated with OOM!")
