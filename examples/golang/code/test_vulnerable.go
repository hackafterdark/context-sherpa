package main

import (
	"context"
	"database/sql"
	"fmt"
)

// Test data structures
type Pet struct {
	Name   string
	Status string
}

func vulnerableCode(db *sql.DB, petID int, pet Pet, categoryIDStr, userName, email string) error {
	ctx := context.Background()

	// This should be caught by our rule
	_, err := db.Exec(fmt.Sprintf(
		"INSERT INTO pets (id, name, category_id, status) VALUES (%d, '%s', %s, '%s')",
		petID, pet.Name, categoryIDStr, pet.Status,
	))
	if err != nil {
		return err
	}

	// This should also be caught
	rows, err := db.Query(fmt.Sprintf("SELECT * FROM users WHERE name = '%s'", userName))
	if err != nil {
		return err
	}
	defer rows.Close()

	// This should also be caught
	row := db.QueryRow(fmt.Sprintf("SELECT id FROM users WHERE email = '%s'", email))

	// This should NOT be caught (safe parameterized query)
	_, err = db.Exec("INSERT INTO pets (id, name) VALUES ($1, $2)", petID, pet.Name)
	return err
}
