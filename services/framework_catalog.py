from __future__ import annotations

from pathlib import Path
from typing import List


_FRAMEWORK_MARKERS = {
    # -------------------------
    # Python web frameworks
    # -------------------------
    "django": ["manage.py", "pyproject.toml", "requirements.txt", "Pipfile", "poetry.lock", "django.cfg"],
    "django_rest_framework": ["manage.py", "rest_framework", "pyproject.toml", "requirements.txt"],
    "wagtail": ["manage.py", "wagtail", "pyproject.toml", "requirements.txt"],
    "flask": ["app.py", "wsgi.py", "requirements.txt", "pyproject.toml", "Pipfile", "poetry.lock"],
    "quart": ["app.py", "quart", "pyproject.toml", "requirements.txt"],
    "fastapi": ["main.py", "pyproject.toml", "requirements.txt", "Pipfile", "poetry.lock"],
    "starlette": ["main.py", "starlette", "pyproject.toml", "requirements.txt"],
    "sanic": ["app.py", "sanic", "pyproject.toml", "requirements.txt"],
    "tornado": ["app.py", "tornado", "pyproject.toml", "requirements.txt"],
    "pyramid": ["development.ini", "production.ini", "setup.py", "pyproject.toml"],
    "bottle": ["app.py", "bottle.py", "requirements.txt"],
    "falcon": ["app.py", "falcon", "pyproject.toml", "requirements.txt"],
    "aiohttp": ["app.py", "aiohttp", "pyproject.toml", "requirements.txt"],
    "streamlit": ["streamlit_app.py", ".streamlit/config.toml", "requirements.txt", "pyproject.toml"],
    "dash": ["app.py", "dash", "requirements.txt", "pyproject.toml"],
    "gradio": ["app.py", "gradio", "requirements.txt", "pyproject.toml"],
    "reflex": ["rxconfig.py", "reflex.json", "pyproject.toml"],
    "celery": ["celery.py", "celeryconfig.py", "pyproject.toml", "requirements.txt"],

    # -------------------------
    # Node.js / JS / TS backends
    # -------------------------
    "node": ["package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "bun.lockb"],
    "express": ["package.json", "app.js", "server.js", "src/app.ts", "src/server.ts"],
    "koa": ["package.json", "koa", "app.js", "src/app.ts"],
    "hapi": ["package.json", "@hapi/hapi", "server.js", "src/server.ts"],
    "fastify": ["package.json", "fastify", "server.js", "src/server.ts"],
    "nestjs": ["nest-cli.json", "package.json", "src/main.ts", "tsconfig.json"],
    "adonisjs": ["adonisrc.ts", "package.json", "start/kernel.ts"],
    "sails": [".sailsrc", "package.json", "config/routes.js"],
    "meteor": [".meteor/packages", ".meteor/release", "package.json"],
    "loopback4": ["package.json", "src/application.ts", "src/index.ts", "lbconfig.json"],
    "strapi": ["package.json", "config/server.js", "src/index.js", "strapi-server.js"],
    "keystone": ["package.json", ".keystone", "keystone.ts", "keystone.js"],
    "feathersjs": ["package.json", "src/app.ts", "config/default.json"],
    "serverless_framework": ["serverless.yml", "serverless.yaml", "serverless.ts", "package.json"],
    "firebase_functions": ["firebase.json", "functions/package.json", "functions/src/index.ts"],

    # Deno / Bun specific
    "deno": ["deno.json", "deno.jsonc", "import_map.json"],
    "deno_fresh": ["fresh.gen.ts", "routes/index.tsx", "deno.json", "import_map.json"],
    "bun": ["bun.lockb", "package.json"],

    # -------------------------
    # PHP backends / CMS
    # -------------------------
    "php_generic": ["composer.json", "composer.lock", "public/index.php"],
    "laravel": ["artisan", "composer.json", "bootstrap/app.php", "config/app.php"],
    "symfony": ["composer.json", "symfony.lock", "bin/console", "config/bundles.php"],
    "lumen": ["artisan", "composer.json", "bootstrap/app.php"],
    "yii2": ["composer.json", "yii", "config/web.php"],
    "codeigniter3": ["application/config/config.php", "system/core/CodeIgniter.php"],
    "codeigniter4": ["app/Config/App.php", "spark", "composer.json"],
    "cakephp": ["bin/cake", "composer.json", "config/app.php"],
    "wordpress": ["wp-config.php", "wp-content", "wp-includes"],
    "drupal": ["core/lib/Drupal.php", "sites/default/settings.php", "composer.json"],
    "joomla": ["configuration.php", "administrator", "libraries/src"],

    # -------------------------
    # Ruby
    # -------------------------
    "ruby": ["Gemfile", "Gemfile.lock"],
    "rails": ["Gemfile", "config/application.rb", "bin/rails", "Rakefile"],
    "sinatra": ["Gemfile", "config.ru", "app.rb"],

    # -------------------------
    # Elixir / Erlang
    # -------------------------
    "elixir": ["mix.exs", "mix.lock"],
    "phoenix": ["mix.exs", "config/config.exs", "lib/*_web.ex", "assets/package.json"],

    # -------------------------
    # Java / Kotlin / Scala
    # -------------------------
    "maven": ["pom.xml", ".mvn/wrapper/maven-wrapper.properties"],
    "gradle": ["build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts", "gradlew"],
    "spring_boot": ["pom.xml", "build.gradle", "src/main/resources/application.yml", "src/main/resources/application.properties"],
    "quarkus": ["pom.xml", "build.gradle", "src/main/resources/application.properties", "src/main/java", ".mvn"],
    "micronaut": ["micronaut-cli.yml", "build.gradle", "pom.xml", "src/main/resources/application.yml"],
    "ktor": ["build.gradle.kts", "settings.gradle.kts", "src/main/kotlin", "resources/application.conf"],
    "play_framework": ["build.sbt", "conf/routes", "conf/application.conf"],
    "vertx": ["pom.xml", "build.gradle", "src/main/java", "src/main/resources"],
    "javalin": ["pom.xml", "build.gradle", "src/main/java", "src/main/kotlin"],

    # -------------------------
    # .NET
    # -------------------------
    "dotnet": [".sln", ".csproj", ".fsproj", ".vbproj", "global.json"],
    "aspnetcore": ["Program.cs", "Startup.cs", "appsettings.json", ".csproj"],
    "blazor": ["Program.cs", "App.razor", "wwwroot/index.html", ".csproj"],
    "maui": ["*.csproj", "Platforms/Android", "Platforms/iOS", "MauiProgram.cs"],

    # -------------------------
    # Go
    # -------------------------
    "go": ["go.mod", "go.sum"],
    "gin": ["go.mod", "main.go", "github.com/gin-gonic/gin"],
    "echo": ["go.mod", "main.go", "github.com/labstack/echo"],
    "fiber": ["go.mod", "main.go", "github.com/gofiber/fiber"],
    "chi": ["go.mod", "main.go", "github.com/go-chi/chi"],

    # -------------------------
    # Rust
    # -------------------------
    "rust": ["Cargo.toml", "Cargo.lock"],
    "actix_web": ["Cargo.toml", "src/main.rs", "actix-web"],
    "axum": ["Cargo.toml", "src/main.rs", "axum"],
    "rocket": ["Cargo.toml", "src/main.rs", "rocket"],
    "warp": ["Cargo.toml", "src/main.rs", "warp"],

    # -------------------------
    # C/C++ / native build systems
    # -------------------------
    "cmake": ["CMakeLists.txt", "cmake/"],
    "meson": ["meson.build", "meson_options.txt"],
    "bazel": ["WORKSPACE", "WORKSPACE.bazel", "BUILD", "BUILD.bazel", "MODULE.bazel"],
    "conan": ["conanfile.txt", "conanfile.py"],
    "vcpkg": ["vcpkg.json", "vcpkg-configuration.json"],

    # -------------------------
    # Frontend frameworks (SPA / meta-frameworks)
    # -------------------------
    "vite": ["vite.config.js", "vite.config.ts", "package.json"],
    "webpack": ["webpack.config.js", "webpack.config.ts", "package.json"],
    "parcel": [".parcelrc", "package.json"],
    "rollup": ["rollup.config.js", "rollup.config.ts", "package.json"],

    "react": ["package.json", "src/index.tsx", "src/index.jsx", "src/main.tsx", "src/main.jsx"],
    "react_router": ["package.json", "src/routes", "src/main.tsx", "react-router"],
    "nextjs": ["next.config.js", "next.config.mjs", "app/page.tsx", "pages/index.tsx", "package.json"],
    "remix": ["remix.config.js", "remix.config.ts", "app/root.tsx", "package.json"],
    "gatsby": ["gatsby-config.js", "gatsby-config.ts", "gatsby-node.js", "package.json"],
    "astro": ["astro.config.mjs", "astro.config.ts", "src/pages", "package.json"],
    "svelte": ["svelte.config.js", "src/App.svelte", "package.json"],
    "sveltekit": ["svelte.config.js", "src/routes/+page.svelte", "package.json"],
    "solidjs": ["package.json", "src/index.tsx", "src/root.tsx"],
    "solidstart": ["package.json", "app/root.tsx", "app/routes"],
    "qwik": ["qwik.config.ts", "src/root.tsx", "package.json"],
    "vue": ["package.json", "src/main.ts", "src/main.js", "vue.config.js"],
    "nuxt": ["nuxt.config.ts", "nuxt.config.js", "package.json", "app.vue"],
    "angular": ["angular.json", "package.json", "tsconfig.app.json", "src/main.ts"],
    "ember": ["ember-cli-build.js", ".ember-cli", "package.json", "app/router.js"],
    "preact": ["package.json", "src/index.tsx", "src/index.jsx", "preact.config.js"],
    "lit": ["package.json", "src/index.ts", "lit.config.js"],
    "alpinejs": ["package.json", "alpinejs", "src/main.js"],
    "htmx": ["package.json", "htmx.min.js", "templates/"],

    # Monorepos / toolchains
    "nx": ["nx.json", "workspace.json", "package.json"],
    "turborepo": ["turbo.json", "package.json"],
    "lerna": ["lerna.json", "package.json"],
    "pnpm_workspace": ["pnpm-workspace.yaml", "package.json"],
    "yarn_workspaces": ["package.json", ".yarnrc.yml", ".yarn/releases"],

    # -------------------------
    # Mobile (hybrid / native)
    # -------------------------
    "kivy": ["main.py", "buildozer.spec", "pyproject.toml", "requirements.txt"],
    "react_native": ["package.json", "metro.config.js", "android/app/build.gradle", "ios/Podfile"],
    "expo": ["app.json", "app.config.js", "app.config.ts", "package.json", "expo"],
    "ionic": ["ionic.config.json", "capacitor.config.ts", "capacitor.config.json", "package.json"],
    "capacitor": ["capacitor.config.ts", "capacitor.config.json", "android/", "ios/"],
    "cordova": ["config.xml", "www/", "hooks/", "package.json"],
    "flutter": ["pubspec.yaml", "android/app/build.gradle", "ios/Runner.xcodeproj", "lib/main.dart"],
    "android_native": ["settings.gradle", "settings.gradle.kts", "app/src/main/AndroidManifest.xml", "gradlew"],
    "ios_native": ["*.xcodeproj", "*.xcworkspace", "Podfile", "Info.plist"],

    # -------------------------
    # Desktop app frameworks
    # -------------------------
    "electron": ["package.json", "electron", "main.js", "main.ts", "electron-builder.yml"],
    "tauri": ["src-tauri/tauri.conf.json", "src-tauri/Cargo.toml", "package.json"],
    "neutralino": ["neutralino.config.json", "resources/"],
    "wails": ["wails.json", "go.mod", "frontend/package.json"],
    "qt_qmake": ["*.pro", "src/main.cpp"],
    "qt_cmake": ["CMakeLists.txt", "src/main.cpp", "Qt6"],
    "gtk": ["meson.build", "src/main.c", "org.gnome.*.desktop"],

    # -------------------------
    # Data / notebooks
    # -------------------------
    "jupyter": ["*.ipynb", ".ipynb_checkpoints/"],
    "jupyterlab": ["jupyter_lab_config.py", "package.json"],
    "dbt": ["dbt_project.yml", "profiles.yml", "models/"],

    # -------------------------
    # Infrastructure / deployment
    # -------------------------
    "docker": ["Dockerfile", "docker-compose.yml", "docker-compose.yaml", ".dockerignore"],
    "kubernetes": ["kustomization.yaml", "helm/Chart.yaml", "charts/Chart.yaml", "deployment.yaml"],
    "helm": ["Chart.yaml", "values.yaml", "templates/"],
    "terraform": ["main.tf", "providers.tf", "variables.tf", "terraform.tfvars", ".terraform.lock.hcl"],
    "pulumi": ["Pulumi.yaml", "Pulumi.dev.yaml", "Pulumi.prod.yaml"],
    "ansible": ["ansible.cfg", "playbook.yml", "inventory", "roles/"],
    "serverless_sam": ["template.yaml", "template.yml", "samconfig.toml"],
    "cdk": ["cdk.json", "bin/*.ts", "lib/*.ts"],
    "cloudformation": ["template.yaml", "template.yml", "*.template"],

    # -------------------------
    # Python packaging / env managers (useful as generic project markers)
    # -------------------------
    "python_poetry": ["pyproject.toml", "poetry.lock"],
    "python_pipenv": ["Pipfile", "Pipfile.lock"],
    "python_setuptools": ["setup.py", "setup.cfg"],
    "python_conda": ["environment.yml", "environment.yaml"],
}



def detect_frameworks(root: Path) -> List[str]:
    root = Path(root)
    hits: List[str] = []
    for name, markers in _FRAMEWORK_MARKERS.items():
        for marker in markers:
            if (root / marker).exists():
                hits.append(name)
                break
    # If package.json exists, hint at JS framework even if markers not found.
    if (root / "package.json").exists():
        if "javascript" not in hits:
            hits.append("javascript")
    return hits


__all__ = ["detect_frameworks"]
