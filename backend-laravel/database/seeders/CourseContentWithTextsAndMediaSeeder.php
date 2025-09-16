<?php

namespace Database\Seeders;

use Illuminate\Database\Console\Seeds\WithoutModelEvents;
use Illuminate\Database\Seeder;
use Illuminate\Support\Facades\DB;
use Carbon\Carbon;

class CourseContentWithTextsAndMediaSeeder extends Seeder
{
    /**
     * Run the database seeds.
     */
    public function run(): void
    {
        $now = Carbon::now();

        // Курсы для заполнения
        $coursesData = [
            [
                'title' => 'Физика для глухих и слабослышащих',
                'slug'  => 'physics-sign',
                'modules' => [
                    'Механика и движение',
                    'Электричество и магнетизм',
                    'Оптика и волны',
                    'Термодинамика',
                ]
            ],
            [
                'title' => 'Математика на жестовом языке',
                'slug'  => 'math-sign',
                'modules' => [
                    'Алгебра и уравнения',
                    'Геометрия и фигуры',
                    'Функции и графики',
                    'Статистика и вероятность',
                ]
            ],
        ];

        foreach ($coursesData as $courseData) {
            // Найти курс по названию
            $course = DB::table('courses')->where('title', $courseData['title'])->first();
            if (!$course) {
                // Создать курс если не существует
                $courseId = DB::table('courses')->insertGetId([
                    'title'           => $courseData['title'],
                    'description'     => 'Курс адаптирован для изучения на жестовом языке с видео-переводом и интерактивными материалами.',
                    'instructor_id'   => 1, // предполагаем что есть инструктор с ID=1
                    'price'           => 2999.00, // 2999 рублей за курс
                    'difficulty_level'=> 'beginner',
                    'duration_hours'  => 40,
                    'is_published'    => 1,
                    'tags'            => json_encode(['физика', 'математика', 'жестовый язык', 'образование'], JSON_UNESCAPED_UNICODE),
                    'created_at'      => $now,
                    'updated_at'      => $now,
                ]);
            } else {
                $courseId = $course->id;
            }

            $slugFolder = $courseData['slug'];
            $withSign3D = true; // включаем 3D жесты

            // Создаём модули и уроки
            foreach ($courseData['modules'] as $i => $modTitle) {
                // 1) Создаём модуль
                $moduleId = DB::table('modules')->insertGetId([
                    'course_id'     => $courseId,
                    'title'         => $modTitle,
                    'description'   => "Модуль посвящён изучению основ раздела '{$modTitle}' с использованием жестового языка и наглядных примеров.",
                    'order_index'   => $i,
                    'is_published'  => 1,
                    'created_at'    => $now,
                    'updated_at'    => $now,
                ]);

                // 2) Создаём уроки в модуле (по 4 урока)
                for ($n = 1; $n <= 4; $n++) {
                    $lessonId = DB::table('lessons')->insertGetId([
                        'module_id'       => $moduleId,
                        'title'           => "{$modTitle} — Урок {$n}",
                        'description'     => "Урок {$n} по теме '{$modTitle}'. Включает теорию, практические задания и видео с жестовым переводом.",
                        'content'         => json_encode($this->makePractice($modTitle, $n), JSON_UNESCAPED_UNICODE),
                        'video_url'       => "/videos/{$slugFolder}/module_".($i+1)."/lesson_{$n}/master.m3u8",
                        'duration_minutes' => rand(7,13),
                        'order_index'     => ($i * 4) + $n - 1,
                        'is_published'    => 1,
                        'created_at'      => $now,
                        'updated_at'      => $now,
                    ]);

                    // 1) текст урока (RU, markdown, основной)
                    DB::table('lesson_texts')->insert([
                        'lesson_id'   => $lessonId,
                        'language'    => 'ru',
                        'format'      => 'markdown',
                        'content'     => $this->makeMarkdownTheory($modTitle, $n),
                        'reading_time'=> 6,
                        'is_primary'  => 1,
                        'meta'        => json_encode(['source'=>'author','v'=>'1.0'], JSON_UNESCAPED_UNICODE),
                        'created_at'  => $now,
                        'updated_at'  => $now,
                    ]);

                    // 2) видео (HLS master — по умолчанию)
                    DB::table('lesson_media')->insert([
                        'lesson_id'    => $lessonId,
                        'type'         => 'video',
                        'provider'     => 'cdn',
                        'storage_path' => "/videos/{$slugFolder}/module_".($i+1)."/lesson_{$n}/master.m3u8",
                        'url'          => "/videos/{$slugFolder}/module_".($i+1)."/lesson_{$n}/master.m3u8",
                        'mime'         => 'application/vnd.apple.mpegurl',
                        'duration'     => rand(7,13) * 60,
                        'quality'      => 'master',
                        'sources'      => json_encode([
                            ['url'=>"/videos/{$slugFolder}/module_".($i+1)."/lesson_{$n}/master.m3u8",'mime'=>'application/vnd.apple.mpegurl','quality'=>'master'],
                        ], JSON_UNESCAPED_UNICODE),
                        'captions'     => json_encode([
                            ['lang'=>'ru','kind'=>'subtitles','url'=>"/captions/{$slugFolder}/m".($i+1)."/l{$n}_ru.vtt"],
                        ], JSON_UNESCAPED_UNICODE),
                        'poster_url'   => "/images/{$slugFolder}/module_".($i+1)."/lesson_{$n}.jpg",
                        'is_default'   => 1,
                        'extra'        => null,
                        'created_at'   => $now,
                        'updated_at'   => $now,
                    ]);

                    // 3) видео (MP4 fallback 720p)
                    DB::table('lesson_media')->insert([
                        'lesson_id'    => $lessonId,
                        'type'         => 'video',
                        'provider'     => 'cdn',
                        'storage_path' => "/videos/{$slugFolder}/module_".($i+1)."/lesson_{$n}/720p.mp4",
                        'url'          => "/videos/{$slugFolder}/module_".($i+1)."/lesson_{$n}/720p.mp4",
                        'mime'         => 'video/mp4',
                        'duration'     => null,
                        'quality'      => '720p',
                        'sources'      => null,
                        'captions'     => null,
                        'poster_url'   => null,
                        'is_default'   => 0,
                        'extra'        => null,
                        'created_at'   => $now,
                        'updated_at'   => $now,
                    ]);

                    // 4) опционально — 3D-жесты (GLB + анимация)
                    if ($withSign3D) {
                        DB::table('lesson_media')->insert([
                            'lesson_id'    => $lessonId,
                            'type'         => 'sign3d',
                            'provider'     => 'cdn',
                            'storage_path' => "/sign3d/{$slugFolder}/module_".($i+1)."/lesson_{$n}/avatar.glb",
                            'url'          => null,
                            'mime'         => 'model/gltf-binary',
                            'duration'     => null,
                            'quality'      => null,
                            'sources'      => null,
                            'captions'     => null,
                            'poster_url'   => null,
                            'is_default'   => 0,
                            'extra'        => json_encode([
                                'glbUrl'   => "/sign3d/{$slugFolder}/module_".($i+1)."/lesson_{$n}/avatar.glb",
                                'animUrl'  => "/sign3d/{$slugFolder}/module_".($i+1)."/lesson_{$n}/gesture.glb",
                                'notes'    => 'DeepMotion export',
                            ], JSON_UNESCAPED_UNICODE),
                            'created_at'   => $now,
                            'updated_at'   => $now,
                        ]);
                    }
                }
            }
        }
    }

    private function makeMarkdownTheory(string $moduleTitle, int $n): string
    {
        return <<<MD
### Теория: {$moduleTitle} — Урок {$n}

В этом уроке мы разбираем ключевые понятия раздела **«{$moduleTitle}»**. 
Материал объясняется простым языком, сопровождается жестовым переводом, наглядными схемами и короткими примерами.

**Сделаем шаги:**
1. Определим важные термины и символы.
2. Посмотрим 2–3 примера (бытовых и учебных).
3. Разберём типичные ошибки и как их избежать.

> Совет: после просмотра видео перечитай конспект и реши 3–5 простых упражнений из практики.

MD;
    }

    private function makePractice(string $moduleTitle, int $n): array
    {
        return [
            [
                'type' => 'task',
                'title'=> "Тренажёр по теме «{$moduleTitle}» (часть {$n})",
                'statement' => 'Выполни 5 коротких заданий. Обращай внимание на шаги решения и единицы измерения.',
                'hints' => ['Сверяйся с конспектом', 'Проверяй итоговую величину/ответ'],
            ],
            [
                'type' => 'mini_project',
                'title'=> 'Мини-проект',
                'statement' => 'Подготовь короткое объяснение жестами реального примера по теме (до 1 минуты).',
            ],
        ];
    }

    private function makeQuiz(string $moduleTitle): array
    {
        return [
            [
                'q' => "Выбери верное утверждение по теме «{$moduleTitle}».",
                'type' => 'single',
                'options' => ['Определение А', 'Определение Б', 'Определение В'],
                'answer'  => 0
            ],
            [
                'q' => 'Какие из перечисленных шагов обязательны?',
                'type' => 'multi',
                'options' => ['Анализ условия', 'Запись формулы', 'Игнорирование единиц'],
                'answers' => [0,1]
            ],
            [
                'q' => 'Короткий ответ: что важно проверить в конце решения?',
                'type' => 'text',
                'answer' => 'Корректность единиц и разумность результата.'
            ]
        ];
    }
}
