from database import DB_PATH, get_connection


def print_table(columns, rows):
    """Print SQLite rows as an aligned text table."""
    headers = [column["name"] for column in columns]
    values = [
        ["" if row[column] is None else str(row[column]) for column in headers]
        for row in rows
    ]
    widths = [
        max([len(header)] + [len(row[index]) for row in values])
        for index, header in enumerate(headers)
    ]
    separator = "-+-".join("-" * width for width in widths)
    print(" | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print(separator)
    for row in values:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


with get_connection(DB_PATH) as connection:

    tables = connection.execute("""
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        ORDER BY name
    """).fetchall()

    print("Available tables:")
    for table in tables:
        print(f"  - {table['name']}")

    for table in ["uploads", "detection_results", "flagged_users", "analysis_history"]:
        print(f"\n{table.upper()}")

        columns = connection.execute(f"PRAGMA table_info({table})").fetchall()

        if not columns:
            print("  Table does not exist.")
            continue

        rows = connection.execute(
            f"SELECT * FROM {table} LIMIT 10"
        ).fetchall()
        if not rows:
            print("  No records yet.")
        else:
            print_table(columns, rows)