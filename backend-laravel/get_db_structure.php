<?php

require_once __DIR__ . '/vendor/autoload.php';

$app = require_once __DIR__ . '/bootstrap/app.php';
$app->make(Illuminate\Contracts\Console\Kernel::class)->bootstrap();

$tables = ['courses', 'lessons', 'modules', 'users', 'enrollments', 'progress', 'reviews', 'ai_chat'];

foreach($tables as $table) {
    echo "\n=== TABLE: $table ===\n";
    try {
        $columns = DB::select('DESCRIBE visual_mind_db.' . $table);
        foreach($columns as $col) {
            echo $col->Field . ' | ' . $col->Type . ' | ' . ($col->Null == 'YES' ? 'NULL' : 'NOT NULL') . ' | ' . ($col->Key ? $col->Key : 'NO KEY') . ' | ' . ($col->Default ?? 'NO DEFAULT') . ' | ' . ($col->Extra ?? 'NO EXTRA') . "\n";
        }
    } catch(Exception $e) {
        echo "Error: " . $e->getMessage() . "\n";
    }
}