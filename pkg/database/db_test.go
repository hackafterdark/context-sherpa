package database

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestInitDB(t *testing.T) {
	// Create a temporary directory for the test database
	tmpDir, err := os.MkdirTemp("", "sherpa-test-*")
	require.NoError(t, err)
	defer os.RemoveAll(tmpDir)

	dbPath := filepath.Join(tmpDir, "test.db")

	t.Run("Initialize new database", func(t *testing.T) {
		db, err := InitDB(dbPath)
		assert.NoError(t, err)
		assert.NotNil(t, db)
		defer db.Close()

		// Verify WAL mode is enabled
		var mode string
		err = db.QueryRow("PRAGMA journal_mode;").Scan(&mode)
		assert.NoError(t, err)
		assert.Equal(t, "wal", mode)

		// Verify directory was created
		_, err = os.Stat(filepath.Dir(dbPath))
		assert.NoError(t, err)
	})

	t.Run("Reopen existing database", func(t *testing.T) {
		db, err := InitDB(dbPath)
		assert.NoError(t, err)
		assert.NotNil(t, db)
		db.Close()
	})

	t.Run("Error on invalid directory path", func(t *testing.T) {
		// Create a file where a directory should be
		invalidDir := filepath.Join(tmpDir, "file_as_dir")
		err := os.WriteFile(invalidDir, []byte("not a dir"), 0644)
		require.NoError(t, err)

		invalidPath := filepath.Join(invalidDir, "test.db")
		db, err := InitDB(invalidPath)
		assert.Error(t, err)
		assert.Nil(t, db)
	})
}
