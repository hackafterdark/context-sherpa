package main

import (
	"context"
	"database/sql"
	"fmt"
)

func vulnerableCode(db *sql.DB) {
	ctx := context.Background()

	// This should be caught by our rule
	_, err := db.Exec(ctx, fmt.Sprintf(
		"INSERT INTO pets (id, name, category_id, status) VALUES (%d, '%s', %s, '%s')",
		petID, pet.Name, categoryIDStr, pet.Status,
	))

	// This should also be caught
	rows, err := db.Query(ctx, fmt.Sprintf("SELECT * FROM users WHERE name = '%s'", userName))

	// This should also be caught
	row := db.QueryRow(ctx, fmt.Sprintf("SELECT id FROM users WHERE email = '%s'", email))

	// This should NOT be caught (safe parameterized query)
	_, err = db.Exec(ctx, "INSERT INTO pets (id, name) VALUES ($1, $2)", petID, pet.Name)

	_ = err
}
