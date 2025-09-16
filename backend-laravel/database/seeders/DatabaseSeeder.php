<?php

namespace Database\Seeders;

use App\Models\User;
// use Illuminate\Database\Console\Seeds\WithoutModelEvents;
use Illuminate\Database\Seeder;

class DatabaseSeeder extends Seeder
{
    /**
     * Seed the application's database.
     */
    public function run(): void
    {
        // User::factory(10)->create();

        // Create test user if not exists
        User::firstOrCreate(
            ['email' => 'test@example.com'],
            [
                'name' => 'Test User',
                'password' => bcrypt('password'),
            ]
        );
        
        // Seed courses, modules, and lessons
        $this->call(CourseSeeder::class);
        $this->call(ModulesAndLessonsSeeder::class);
        $this->call(PhysicsCourseSeeder::class);
        $this->call(CourseContentWithTextsAndMediaSeeder::class);
        
        // Seed purchases (user enrollments)
        $this->call(PurchaseSeeder::class);
    }
}
