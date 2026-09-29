#!/usr/bin/env python3
"""
update_changelog.py - Helper for updating changelog

Automates creation of entries in history/changelog.md based on git commits.

Usage:
    python llm/tools/update_changelog.py
    python llm/tools/update_changelog.py --since 2026-03-19
    python llm/tools/update_changelog.py --help
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import re


class ChangelogUpdater:
    def __init__(self, repo_root: Optional[Path] = None):
        self.repo_root = repo_root or Path(__file__).parent.parent.parent
        self.changelog_path = self.repo_root / "llm" / "history" / "changelog.md"

    def get_git_commits(self, since: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Get list of git commits since given date.

        Returns:
            List of dicts with keys: hash, date, message, body
        """
        cmd = ["git", "log", "--pretty=format:%H|%ai|%s|%b", "--no-merges"]

        if since:
            cmd.append(f"--since={since}")
        else:
            # If since not specified - use last date from changelog
            last_date = self._get_last_changelog_date()
            if last_date:
                cmd.append(f"--since={last_date}")

        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Error running git log: {e}")
            return []

        commits = []
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue

            parts = line.split('|', 3)
            if len(parts) < 3:
                continue

            commit_hash, date, subject = parts[:3]
            body = parts[3] if len(parts) > 3 else ""

            commits.append({
                'hash': commit_hash[:7],  # Short hash
                'date': date.split()[0],  # YYYY-MM-DD
                'subject': subject.strip(),
                'body': body.strip()
            })

        return commits

    def _get_last_changelog_date(self) -> Optional[str]:
        """Read last date from changelog."""
        if not self.changelog_path.exists():
            return None

        with open(self.changelog_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Search for lines like "## YYYY-MM-DD - ..."
        matches = re.findall(r'## (\d{4}-\d{2}-\d{2}) -', content)
        if matches:
            return matches[0]  # First (most recent) date

        return None

    def group_commits_by_topic(self, commits: List[Dict[str, str]]) -> Dict[str, List[Dict]]:
        """
        Group commits by topic based on message prefix.

        Example prefixes:
        - [Zoom]: ...
        - [Refactor]: ...
        - [Docs]: ...
        - [LLM]: ...
        """
        groups = {}

        for commit in commits:
            subject = commit['subject']

            # Try to extract topic from [Topic]: format
            match = re.match(r'\[([^\]]+)\]:\s*(.*)', subject)
            if match:
                topic = match.group(1)
                description = match.group(2)
            else:
                # Fallback: use first word or "Other"
                words = subject.split()
                topic = words[0] if words else "Other"
                description = subject

            if topic not in groups:
                groups[topic] = []

            groups[topic].append({
                **commit,
                'description': description
            })

        return groups

    def generate_changelog_entry(
        self,
        date: str,
        commits: List[Dict[str, str]]
    ) -> str:
        """
        Generate changelog entry.

        Args:
            date: Date in YYYY-MM-DD format
            commits: List of commits

        Returns:
            Markdown text of entry
        """
        # Group by topics
        groups = self.group_commits_by_topic(commits)

        # Determine main topic (most frequent)
        if len(groups) == 1:
            main_topic = list(groups.keys())[0]
        else:
            # Multiple topics - use "Mixed changes"
            main_topic = "Mixed changes"

        # Start entry
        entry = f"## {date} - {main_topic}\n\n"
        entry += "**Что сделано:**\n"

        # List changes by topic
        for topic, topic_commits in sorted(groups.items()):
            if len(groups) > 1:
                entry += f"\n_{topic}_:\n"

            for commit in topic_commits:
                entry += f"- {commit['description']}\n"

        entry += "\n**Технические детали:**\n\n"
        entry += "(Agent: add details about changes here)\n"

        entry += "\n**Git commits:**"
        for commit in commits:
            entry += f" `{commit['hash']}`"
        entry += "\n"

        entry += "\n**Участники:** (Agent: specify who worked on this)\n"

        entry += "\n---\n"

        return entry

    def append_to_changelog(self, entry: str) -> None:
        """Add entry to beginning of changelog (after header)."""
        if not self.changelog_path.exists():
            print(f"Error: {self.changelog_path} not found")
            return

        with open(self.changelog_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find end of header (after "---")
        lines = content.split('\n')
        insert_index = 0
        for i, line in enumerate(lines):
            if line.strip() == '---':
                insert_index = i + 1
                break

        # Insert new entry
        lines.insert(insert_index, '\n' + entry)

        with open(self.changelog_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"✅ Entry added to {self.changelog_path}")

    def run(self, since: Optional[str] = None, dry_run: bool = False) -> None:
        """
        Main method - run changelog update.

        Args:
            since: Date to start from (YYYY-MM-DD)
            dry_run: If True - only show what would be added
        """
        print(f"🔍 Searching for commits since: {since or 'last changelog date'}...")

        commits = self.get_git_commits(since)

        if not commits:
            print("ℹ️  No new commits found")
            return

        print(f"📝 Found {len(commits)} commit(s)")

        # Group by date
        by_date = {}
        for commit in commits:
            date = commit['date']
            if date not in by_date:
                by_date[date] = []
            by_date[date].append(commit)

        # Generate entries for each date
        for date in sorted(by_date.keys(), reverse=True):
            entry = self.generate_changelog_entry(date, by_date[date])

            print(f"\n{'='*60}")
            print(f"Generated entry for {date}:")
            print('='*60)
            print(entry)

            if dry_run:
                print("(Dry run - not actually adding to changelog)")
            else:
                self.append_to_changelog(entry)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Update changelog from git commits"
    )
    parser.add_argument(
        '--since',
        help='Date to start from (YYYY-MM-DD)',
        default=None
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be added without actually modifying changelog'
    )

    args = parser.parse_args()

    updater = ChangelogUpdater()
    updater.run(since=args.since, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
