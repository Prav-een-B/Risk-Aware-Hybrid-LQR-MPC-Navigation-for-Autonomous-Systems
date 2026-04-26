import markdown
import codecs

# Read markdown
with codecs.open('Adaptive_MPC_Theory.md', mode='r', encoding='utf-8') as f:
    text = f.read()

# Convert to html
html_body = markdown.markdown(text, extensions=['fenced_code', 'tables'])

# Template with beautiful CSS and MathJax
html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Adaptive MPC Theory</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        body {{
            font-family: 'Inter', sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 40px;
        }}
        h1, h2, h3, h4 {{
            color: #1a1a1a;
            margin-top: 2em;
            margin-bottom: 0.5em;
        }}
        h1 {{
            border-bottom: 2px solid #eaecef;
            padding-bottom: 0.3em;
            font-size: 2.2em;
        }}
        h2 {{
            border-bottom: 1px solid #eaecef;
            padding-bottom: 0.3em;
            font-size: 1.8em;
        }}
        p {{
            margin-bottom: 1.2em;
            font-size: 1.05em;
        }}
        code {{
            background-color: #f6f8fa;
            padding: 0.2em 0.4em;
            border-radius: 3px;
            font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;
            font-size: 85%;
        }}
        pre {{
            background-color: #f6f8fa;
            padding: 16px;
            border-radius: 6px;
            overflow: auto;
        }}
        pre code {{
            background-color: transparent;
            padding: 0;
            font-size: 90%;
        }}
        .math {{
            overflow-x: auto;
            overflow-y: hidden;
            padding: 0.5em 0;
        }}
    </style>
    <script>
        MathJax = {{
          tex: {{
            inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
            displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
            processEscapes: true
          }},
          svg: {{
            fontCache: 'global'
          }}
        }};
    </script>
    <script type="text/javascript" id="MathJax-script" async
      src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js">
    </script>
</head>
<body>
    {html_body}
    
    <!-- We add a script to signal when MathJax is done rendering, which can be useful for headless browsers -->
    <script>
        MathJax.startup.promise.then(() => {{
            const div = document.createElement('div');
            div.id = 'mathjax-done';
            document.body.appendChild(div);
        }});
    </script>
</body>
</html>
"""

with codecs.open('Adaptive_MPC_Theory.html', mode='w', encoding='utf-8') as f:
    f.write(html_template)

print("Successfully generated Adaptive_MPC_Theory.html")
