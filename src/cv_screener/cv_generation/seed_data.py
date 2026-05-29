"""Deterministic seed data for local CV content generation."""

SEED_PROFILES: list[dict] = [
    {
        "full_name": "Marta Alvarez",
        "email": "marta.alvarez@example.com",
        "phone": "+34 600 111 111",
        "location": "Barcelona, Spain",
        "professional_summary": "Backend engineer with experience building APIs, data pipelines, and internal tooling.",
        "skills": ["Python", "FastAPI", "PostgreSQL", "Docker", "AWS"],
        "experience": [
            {
                "company": "NovaStack",
                "role": "Backend Engineer",
                "start_date": "2021-03-01",
                "end_date": None,
                "summary": "Built backend services for workflow automation.",
                "highlights": [
                    "Implemented REST APIs for internal products.",
                    "Reduced batch processing time by optimizing SQL queries.",
                ],
                "technologies": ["Python", "FastAPI", "PostgreSQL"],
            }
        ],
        "education": [
            {
                "institution": "Universitat Politecnica de Catalunya",
                "degree": "BSc",
                "field_of_study": "Computer Engineering",
                "graduation_year": 2020,
            }
        ],
    },
    {
        "full_name": "Daniel Romero",
        "email": "daniel.romero@example.com",
        "phone": "+34 600 222 222",
        "location": "Madrid, Spain",
        "professional_summary": "Full-stack developer focused on product delivery, frontend quality, and developer experience.",
        "skills": ["TypeScript", "React", "Node.js", "GraphQL", "Playwright"],
        "experience": [
            {
                "company": "PixelForge",
                "role": "Full-stack Developer",
                "start_date": "2020-06-01",
                "end_date": None,
                "summary": "Delivered customer-facing features across web applications.",
                "highlights": [
                    "Shipped React interfaces for analytics workflows.",
                    "Added end-to-end test coverage for release-critical flows.",
                ],
                "technologies": ["TypeScript", "React", "Node.js"],
            }
        ],
        "education": [
            {
                "institution": "Universidad Carlos III de Madrid",
                "degree": "BEng",
                "field_of_study": "Software Engineering",
                "graduation_year": 2019,
            }
        ],
    },
    {
        "full_name": "Lucia Moreno",
        "email": "lucia.moreno@example.com",
        "phone": "+34 600 333 333",
        "location": "Valencia, Spain",
        "professional_summary": "Data engineer with experience in ETL pipelines, analytics modeling, and platform reliability.",
        "skills": ["Python", "dbt", "Airflow", "BigQuery", "Terraform"],
        "experience": [
            {
                "company": "SignalRiver",
                "role": "Data Engineer",
                "start_date": "2022-01-10",
                "end_date": None,
                "summary": "Maintained data ingestion and transformation pipelines for analytics teams.",
                "highlights": [
                    "Modeled warehouse tables for self-serve reporting.",
                    "Improved observability for scheduled data jobs.",
                ],
                "technologies": ["Python", "Airflow", "BigQuery"],
            }
        ],
        "education": [
            {
                "institution": "Universitat de Valencia",
                "degree": "MSc",
                "field_of_study": "Data Science",
                "graduation_year": 2021,
            }
        ],
    },
]
