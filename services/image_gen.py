"""
Image & Graph Generation Service
==================================
Generates:
  1. AI Photos/Images using Pollinations.ai (free, no API key)
  2. Charts/Graphs using matplotlib (bar, line, pie, scatter, histogram)
  3. Diagrams using matplotlib (flowcharts, comparisons)
  4. Data visualizations from document context
  5. Text-to-image descriptions rendered as styled info-graphics

The service generates images server-side and returns them as base64 or
serves them via a static endpoint.
"""

import os
import io
import json
import uuid
import base64
import re
import textwrap
import urllib.parse
import urllib.request
from typing import Optional, Dict, Any, List, Tuple

# ── Chart generation with matplotlib ─────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Directory to store generated images
IMAGE_DIR = os.path.join(os.path.dirname(__file__), "..", "generated_images")
os.makedirs(IMAGE_DIR, exist_ok=True)

# Load GEMINI_API_KEY from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def _try_openai_dalle(prompt: str, width: int, height: int) -> Optional[bytes]:
    """Generate image via OpenAI DALL·E API (fallback)."""
    if not OPENAI_API_KEY:
        print("[ImageGen] OpenAI DALL·E failed: OPENAI_API_KEY not configured")
        return None
    try:
        import urllib.request, json
        url = "https://api.openai.com/v1/images/generations"
        payload = {
            "prompt": prompt,
            "n": 1,
            "size": f"{width}x{height}" if width == height else "1024x1024"
        }
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }
        )
        with urllib.request.urlopen(req, timeout=20) as response:
            res = json.loads(response.read().decode('utf-8'))
        # Expect res['data'][0]['url']
        image_url = res.get('data', [{}])[0].get('url')
        if image_url:
            return _download_image(image_url, timeout=10)
    except Exception as e:
        print(f"[ImageGen] OpenAI DALL·E failed: {e}")
    return None


# ── Theme Configuration ──────────────────────────────────────────
DARK_THEME = {
    "bg": "#0f0f1a",
    "card_bg": "#14142b",
    "text": "#f1f1f8",
    "subtext": "#9b9bc4",
    "grid": "#1e1e3a",
    "accent_colors": ["#818cf8", "#c084fc", "#22d3a5", "#fbbf24", "#f87171", "#60a5fa", "#fb923c", "#a78bfa"],
    "gradient_start": "#818cf8",
    "gradient_end": "#c084fc",
}


def _setup_dark_style():
    """Apply the dark theme to matplotlib."""
    plt.rcParams.update({
        'figure.facecolor': DARK_THEME['bg'],
        'axes.facecolor': DARK_THEME['card_bg'],
        'axes.edgecolor': DARK_THEME['grid'],
        'axes.labelcolor': DARK_THEME['text'],
        'text.color': DARK_THEME['text'],
        'xtick.color': DARK_THEME['subtext'],
        'ytick.color': DARK_THEME['subtext'],
        'grid.color': DARK_THEME['grid'],
        'grid.alpha': 0.3,
        'font.family': 'sans-serif',
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.titleweight': 'bold',
        'legend.facecolor': DARK_THEME['card_bg'],
        'legend.edgecolor': DARK_THEME['grid'],
        'legend.fontsize': 10,
    })


def generate_bar_chart(
    title: str,
    labels: List[str],
    values: List[float],
    ylabel: str = "Value",
    xlabel: str = "",
    horizontal: bool = False
) -> str:
    """Generate a styled bar chart and return the image ID."""
    _setup_dark_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = DARK_THEME['accent_colors'][:len(labels)]

    if horizontal:
        bars = ax.barh(labels, values, color=colors, edgecolor='none', height=0.6)
        ax.set_xlabel(ylabel)
        if xlabel:
            ax.set_ylabel(xlabel)
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + max(values) * 0.02, bar.get_y() + bar.get_height() / 2,
                    f'{val:,.1f}' if isinstance(val, float) else f'{val:,}',
                    va='center', fontsize=10, color=DARK_THEME['subtext'])
    else:
        bars = ax.bar(labels, values, color=colors, edgecolor='none', width=0.6, zorder=3)
        ax.set_ylabel(ylabel)
        if xlabel:
            ax.set_xlabel(xlabel)
        # Add value labels on top
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02,
                    f'{val:,.1f}' if isinstance(val, float) else f'{val:,}',
                    ha='center', fontsize=10, color=DARK_THEME['subtext'])

    ax.set_title(title, pad=20, fontsize=16, fontweight='bold')
    ax.grid(axis='y' if not horizontal else 'x', alpha=0.15, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Rotate x labels if too many
    if not horizontal and len(labels) > 5:
        plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    return _save_figure(fig)


def generate_line_chart(
    title: str,
    x_data: List,
    y_datasets: List[Dict[str, Any]],
    xlabel: str = "",
    ylabel: str = "Value"
) -> str:
    """
    Generate a multi-line chart.
    y_datasets: [{"label": "Series 1", "values": [...]}, ...]
    """
    _setup_dark_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = DARK_THEME['accent_colors']
    for i, dataset in enumerate(y_datasets):
        color = colors[i % len(colors)]
        ax.plot(x_data, dataset['values'], marker='o', markersize=5,
                linewidth=2.5, label=dataset.get('label', f'Series {i+1}'),
                color=color, zorder=3)
        # Add a subtle fill below the line
        ax.fill_between(x_data, dataset['values'], alpha=0.05, color=color)

    ax.set_title(title, pad=20, fontsize=16, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.15, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if len(y_datasets) > 1:
        ax.legend(framealpha=0.8)

    plt.tight_layout()
    return _save_figure(fig)


def generate_pie_chart(
    title: str,
    labels: List[str],
    values: List[float],
    show_percentage: bool = True
) -> str:
    """Generate a styled pie/donut chart."""
    _setup_dark_style()
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = DARK_THEME['accent_colors'][:len(labels)]

    # Create a donut chart (more modern)
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, colors=colors,
        autopct='%1.1f%%' if show_percentage else '',
        pctdistance=0.8, startangle=140,
        wedgeprops=dict(width=0.5, edgecolor=DARK_THEME['bg'], linewidth=2)
    )

    # Style the text
    for text in texts:
        text.set_color(DARK_THEME['text'])
        text.set_fontsize(11)
    for autotext in autotexts:
        autotext.set_color(DARK_THEME['text'])
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')

    ax.set_title(title, pad=25, fontsize=16, fontweight='bold')

    plt.tight_layout()
    return _save_figure(fig)


def generate_scatter_plot(
    title: str,
    x_data: List[float],
    y_data: List[float],
    labels: Optional[List[str]] = None,
    xlabel: str = "X",
    ylabel: str = "Y",
    sizes: Optional[List[float]] = None
) -> str:
    """Generate a scatter plot."""
    _setup_dark_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    s = sizes if sizes else [80] * len(x_data)
    scatter = ax.scatter(x_data, y_data, s=s, c=DARK_THEME['accent_colors'][0],
                         alpha=0.7, edgecolors='white', linewidth=0.5, zorder=3)

    if labels:
        for i, label in enumerate(labels):
            ax.annotate(label, (x_data[i], y_data[i]),
                        textcoords="offset points", xytext=(8, 8),
                        fontsize=9, color=DARK_THEME['subtext'])

    ax.set_title(title, pad=20, fontsize=16, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.15, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return _save_figure(fig)


def generate_histogram(
    title: str,
    data: List[float],
    bins: int = 20,
    xlabel: str = "Value",
    ylabel: str = "Frequency"
) -> str:
    """Generate a histogram."""
    _setup_dark_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    n, bins_arr, patches = ax.hist(data, bins=bins, color=DARK_THEME['accent_colors'][0],
                                   edgecolor=DARK_THEME['bg'], linewidth=1, alpha=0.85, zorder=3)

    # Color gradient on bars
    cm = plt.cm.get_cmap('cool')
    max_val = max(n)
    for count, patch in zip(n, patches):
        color_idx = count / max_val if max_val > 0 else 0
        patch.set_facecolor(cm(color_idx * 0.7 + 0.15))

    ax.set_title(title, pad=20, fontsize=16, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=0.15, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return _save_figure(fig)


def generate_comparison_chart(
    title: str,
    categories: List[str],
    group_names: List[str],
    group_values: List[List[float]],
    ylabel: str = "Value"
) -> str:
    """Generate a grouped bar chart for comparisons."""
    _setup_dark_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(categories))
    n_groups = len(group_names)
    width = 0.7 / n_groups
    colors = DARK_THEME['accent_colors']

    for i, (name, vals) in enumerate(zip(group_names, group_values)):
        offset = (i - n_groups / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=name,
                      color=colors[i % len(colors)], edgecolor='none', zorder=3)

    ax.set_title(title, pad=20, fontsize=16, fontweight='bold')
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(framealpha=0.8)
    ax.grid(axis='y', alpha=0.15, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if len(categories) > 5:
        plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    return _save_figure(fig)


def generate_infographic(
    title: str,
    items: List[Dict[str, str]],
    style: str = "cards"
) -> str:
    """
    Generate a styled infographic with key facts/stats.
    items: [{"label": "...", "value": "...", "icon": "emoji"}, ...]
    """
    _setup_dark_style()

    n = len(items)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    fig_width = max(10, cols * 3)
    fig_height = max(4, rows * 2.5 + 2)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

    # Ensure axes is 2D
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle(title, fontsize=18, fontweight='bold', color=DARK_THEME['text'], y=0.98)

    colors = DARK_THEME['accent_colors']

    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        ax.axis('off')

        if idx < n:
            item = items[idx]
            color = colors[idx % len(colors)]

            # Draw card background
            card = FancyBboxPatch((0.05, 0.05), 0.9, 0.9,
                                  boxstyle="round,pad=0.1",
                                  facecolor=DARK_THEME['card_bg'],
                                  edgecolor=color, linewidth=1.5,
                                  transform=ax.transAxes)
            ax.add_patch(card)

            # Icon/emoji
            icon = item.get('icon', '📊')
            ax.text(0.5, 0.72, icon, fontsize=26, ha='center', va='center',
                    transform=ax.transAxes)

            # Value
            ax.text(0.5, 0.45, str(item.get('value', '')),
                    fontsize=16, fontweight='bold', ha='center', va='center',
                    color=color, transform=ax.transAxes)

            # Label
            label = textwrap.fill(item.get('label', ''), width=15)
            ax.text(0.5, 0.2, label,
                    fontsize=9, ha='center', va='center',
                    color=DARK_THEME['subtext'], transform=ax.transAxes)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return _save_figure(fig)


def generate_timeline(
    title: str,
    events: List[Dict[str, str]]
) -> str:
    """
    Generate a timeline visualization.
    events: [{"date": "2024", "label": "Event name", "desc": "optional description"}, ...]
    """
    _setup_dark_style()

    n = len(events)
    fig, ax = plt.subplots(figsize=(14, max(4, n * 0.8)))

    colors = DARK_THEME['accent_colors']
    y_positions = list(range(n))

    # Draw timeline line
    ax.plot([0.35, 0.35], [-0.5, n - 0.5], color=DARK_THEME['grid'], linewidth=2, zorder=1)

    for i, event in enumerate(events):
        color = colors[i % len(colors)]
        y = n - 1 - i

        # Draw dot
        ax.scatter(0.35, y, s=120, color=color, zorder=3, edgecolors='white', linewidth=1.5)

        # Date on the left
        ax.text(0.3, y, event.get('date', ''), fontsize=11, fontweight='bold',
                ha='right', va='center', color=color)

        # Event label on the right
        label = event.get('label', '')
        desc = event.get('desc', '')
        ax.text(0.42, y + 0.05, label, fontsize=12, fontweight='bold',
                ha='left', va='center', color=DARK_THEME['text'])
        if desc:
            ax.text(0.42, y - 0.2, textwrap.fill(desc, width=60),
                    fontsize=9, ha='left', va='center', color=DARK_THEME['subtext'])

    ax.set_xlim(0, 1.2)
    ax.set_ylim(-0.8, n - 0.2)
    ax.axis('off')
    ax.set_title(title, pad=20, fontsize=16, fontweight='bold')

    plt.tight_layout()
    return _save_figure(fig)


def _save_figure(fig) -> str:
    """Save a matplotlib figure and return the image ID."""
    image_id = str(uuid.uuid4())
    filepath = os.path.join(IMAGE_DIR, f"{image_id}.png")

    fig.savefig(filepath, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)

    return image_id


def get_image_path(image_id: str) -> Optional[str]:
    """Get the file path for a generated image."""
    filepath = os.path.join(IMAGE_DIR, f"{image_id}.png")
    if os.path.exists(filepath):
        return filepath
    return None


def get_image_base64(image_id: str) -> Optional[str]:
    """Get base64-encoded image data."""
    filepath = get_image_path(image_id)
    if not filepath:
        return None
    with open(filepath, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


# ── AI Photo Generation (Multi-Provider with Fallbacks) ───────────

def _download_image(url: str, timeout: int = 5) -> Optional[bytes]:
    """Download image bytes from a URL with SSL bypass for Windows and a fast timeout."""
    import ssl
    ctx = ssl._create_unverified_context()
    req = urllib.request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    with urllib.request.urlopen(req, context=ctx, timeout=timeout) as response:
        return response.read()


def _try_google_imagen(prompt: str, width: int, height: int) -> Optional[bytes]:
    """Generate image using Google's Imagen 3 API in Google AI Studio using the user's GEMINI_API_KEY."""
    if not GEMINI_API_KEY:
        print("[ImageGen] Google Imagen failed: GEMINI_API_KEY not configured")
        return None
        
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-generate-002:predict?key={GEMINI_API_KEY}"
        print(f"[ImageGen] Generating image via Google Imagen 3: {prompt[:80]}...")
        
        # Determine best aspect ratio based on width and height
        # Imagen 3 supports: "1:1", "3:4", "4:3", "9:16", "16:9"
        aspect_ratio = "1:1"
        ratio = width / height
        if abs(ratio - 1.0) < 0.15:
            aspect_ratio = "1:1"
        elif abs(ratio - (16/9)) < 0.2:
            aspect_ratio = "16:9"
        elif abs(ratio - (9/16)) < 0.2:
            aspect_ratio = "9:16"
        elif abs(ratio - (4/3)) < 0.2:
            aspect_ratio = "4:3"
        elif abs(ratio - (3/4)) < 0.2:
            aspect_ratio = "3:4"
            
        payload = {
            "instances": [
                {
                    "prompt": prompt
                }
            ],
            "parameters": {
                "sampleCount": 1,
                "aspectRatio": aspect_ratio,
                "outputMimeType": "image/png"
            }
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        
        import ssl
        ctx = ssl._create_unverified_context()
        
        # Google AI Studio can take up to 10 seconds for image generation
        with urllib.request.urlopen(req, context=ctx, timeout=20) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            
        predictions = res_data.get("predictions", [])
        if predictions:
            img_b64 = predictions[0].get("bytesBase64Encoded")
            if img_b64:
                return base64.b64decode(img_b64)
                
        print(f"[ImageGen] Google Imagen response format mismatch or empty predictions")
    except Exception as e:
        print(f"[ImageGen] Google Imagen failed: {e}")
        
    return None


def _try_pollinations(prompt: str, width: int, height: int) -> Optional[bytes]:
    """Try Pollinations.ai image generation (5s timeout)."""
    encoded_prompt = urllib.parse.quote(prompt)
    seed = uuid.uuid4().int % 100000
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={width}&height={height}&nologo=true&seed={seed}"
    print(f"[ImageGen] Trying Pollinations.ai...")
    return _download_image(url, timeout=5)


def _try_pollinations_v2(prompt: str, width: int, height: int) -> Optional[bytes]:
    """Try Pollinations.ai with model parameter (5s timeout)."""
    encoded_prompt = urllib.parse.quote(prompt)
    seed = uuid.uuid4().int % 100000
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={width}&height={height}&nologo=true&seed={seed}&model=flux"
    print(f"[ImageGen] Trying Pollinations.ai v2 (flux model)...")
    return _download_image(url, timeout=5)


def _try_airforce(prompt: str, width: int, height: int) -> Optional[bytes]:
    """Try AirForce free image API (5s timeout)."""
    encoded_prompt = urllib.parse.quote(prompt)
    url = f"https://api.airforce/v1/imagine2?prompt={encoded_prompt}"
    print(f"[ImageGen] Trying AirForce API...")
    return _download_image(url, timeout=5)


def _try_hercai(prompt: str, width: int, height: int) -> Optional[bytes]:
    """Try HercAI free image API (v3/text2image) (5s timeout)."""
    try:
        encoded_prompt = urllib.parse.quote(prompt)
        url = f"https://hercai.onrender.com/v3/text2image?prompt={encoded_prompt}"
        print(f"[ImageGen] Trying HercAI API...")
        import ssl
        ctx = ssl._create_unverified_context()
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        })
        with urllib.request.urlopen(req, context=ctx, timeout=5) as response:
            data = json.loads(response.read().decode('utf-8'))
        image_url = data.get('url', '')
        if image_url:
            return _download_image(image_url, timeout=5)
    except Exception as e:
        print(f"[ImageGen] HercAI parse/fetch failed: {e}")
    return None


def _try_robohash(prompt: str, width: int, height: int) -> Optional[bytes]:
    """Try Robohash robot/avatar generator (highly reliable, no API key, fast, never blocks)."""
    encoded_prompt = urllib.parse.quote(prompt)
    url = f"https://robohash.org/{encoded_prompt}.png?size={width}x{height}"
    print(f"[ImageGen] Trying Robohash (kitten set)...")
    return _download_image(url, timeout=5)


def _try_picsum_placeholder(prompt: str, width: int, height: int) -> Optional[bytes]:
    """Get a random photo from Lorem Picsum (5s timeout) with cache-busting."""
    url = f"https://picsum.photos/{width}/{height}?random={uuid.uuid4()}"
    print(f"[ImageGen] Trying Lorem Picsum...")
    return _download_image(url, timeout=5)


def generate_local_placeholder(prompt: str, width: int = 800, height: int = 800) -> Optional[bytes]:
    """
    Generate an abstract digital poster with the prompt text using matplotlib.
    This works entirely offline and acts as the ultimate local fallback.
    """
    if not HAS_MATPLOTLIB:
        return None
    try:
        print(f"[ImageGen] Generating local matplotlib placeholder...")
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Setup modern dark theme figure
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        fig.patch.set_facecolor('#0f0f1a')
        ax.set_facecolor('#0f0f1a')
        
        # Draw a colorful neon background pattern
        x = np.linspace(0, 10, 100)
        y = np.linspace(0, 10, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X/2.5) * np.cos(Y/2.5) + np.sin(Y/1.8)
        ax.imshow(Z, cmap='magma', extent=[0, 10, 0, 10], origin='lower', alpha=0.35)
        
        # Add abstract glowing particles
        from matplotlib.patches import Circle, Rectangle
        for _ in range(8):
            cx, cy = np.random.uniform(1.5, 8.5, 2)
            r = np.random.uniform(0.4, 1.8)
            color = np.random.choice(['#818cf8', '#c084fc', '#22d3a5', '#fbbf24', '#f87171'])
            circle = Circle((cx, cy), r, color=color, alpha=0.18, transform=ax.transData)
            ax.add_patch(circle)
            
        # Draw sleek double border
        ax.add_patch(Rectangle((0.15, 0.15), 9.7, 9.7, fill=False, edgecolor='#312e81', linewidth=1.5))
        ax.add_patch(Rectangle((0.3, 0.3), 9.4, 9.4, fill=False, edgecolor='#1e1e3a', linewidth=2.5))
        
        # Title
        ax.text(5, 7.8, "AI IMAGE GENERATOR", fontsize=20, color='#818cf8', 
                weight='bold', ha='center', va='center', letterspacing=2)
        ax.text(5, 7.1, "[ Offline Local Generation Mode ]", fontsize=10, color='#9b9bc4', 
                style='italic', ha='center', va='center')
        
        # Wrap prompt text nicely
        import textwrap
        wrapped_prompt = "\n".join(textwrap.wrap(prompt, width=38))
        
        # Central prompt block
        ax.text(5, 4.4, wrapped_prompt, fontsize=13, color='#f1f1f8', 
                weight='medium', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='#14142b', edgecolor='#c084fc', alpha=0.92, linewidth=1.5))
        
        # Footer
        ax.text(5, 1.3, "RAG AI Agent • Graphics Engine", fontsize=9, color='#5c5c8a', 
                ha='center', va='center')
        
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, facecolor=fig.get_facecolor())
        plt.close(fig)
        return buf.getvalue()
    except Exception as e:
        print(f"[ImageGen] Local placeholder failed: {e}")
        return None


def generate_ai_photo(prompt: str, width: int = 1024, height: int = 1024) -> Optional[str]:
    """
    Generate an AI photo using free image generation APIs.
    Tries multiple providers with fast timeouts and falls back to local generation:
      1. Google Imagen 3 (primary, using user's GEMINI_API_KEY)
      2. Pollinations.ai
      3. Pollinations.ai v2 with flux model
      4. AirForce API
      5. HercAI API
      
      7. Lorem Picsum (random placeholder with cache busting)
      8. Local Matplotlib Abstract Card (guaranteed offline success)
    Downloads the generated image and saves it locally.
    Returns image_id on success, None on failure.
    """
    providers = [
        ("Google-Imagen3", lambda: _try_google_imagen(prompt, width, height)),
        ("OpenAI-DALL·E", lambda: _try_openai_dalle(prompt, width, height)),
        ("Pollinations", lambda: _try_pollinations(prompt, width, height)),
        ("Pollinations-v2", lambda: _try_pollinations_v2(prompt, width, height)),
        ("AirForce", lambda: _try_airforce(prompt, width, height)),
        ("HercAI", lambda: _try_hercai(prompt, width, height)),
        ("Picsum", lambda: _try_picsum_placeholder(prompt, min(width, 800), min(height, 800))),
        ("Local-Matplotlib", lambda: generate_local_placeholder(prompt, min(width, 800), min(height, 800)))
    ]


    print(f"[ImageGen] Generating AI photo: {prompt[:80]}...")

    last_error = None
    for name, provider_fn in providers:
        try:
            image_data = provider_fn()
            if image_data and len(image_data) >= 1000:
                # Save to disk
                image_id = str(uuid.uuid4())
                filepath = os.path.join(IMAGE_DIR, f"{image_id}.png")
                with open(filepath, 'wb') as f:
                    f.write(image_data)
                print(f"[ImageGen] AI photo saved via {name}: {image_id} ({len(image_data)} bytes)")
                return image_id
            else:
                size = len(image_data) if image_data else 0
                print(f"[ImageGen] {name} returned too-small response ({size} bytes), trying next...")
        except Exception as e:
            last_error = e
            print(f"[ImageGen] {name} failed: {e}")
            continue

    # If absolutely all of them failed
    import traceback
    error_msg = f"All image providers and local placeholders failed. Last error: {last_error}\n{traceback.format_exc() if last_error else 'N/A'}"
    print(f"[ImageGen] {error_msg}")
    try:
        with open(os.path.join(os.path.dirname(__file__), "..", "image_gen_error.log"), "w") as f:
            f.write(error_msg)
    except Exception:
        pass
    return None


# ── Chart type detection and routing ─────────────────────────────

CHART_PATTERNS = {
    'bar_chart': [
        r'bar\s*chart', r'bar\s*graph', r'column\s*chart',
        r'compare.*(?:values|numbers|stats)', r'show.*distribution',
    ],
    'line_chart': [
        r'line\s*chart', r'line\s*graph', r'trend', r'over\s*time',
        r'time\s*series', r'growth', r'progression',
    ],
    'pie_chart': [
        r'pie\s*chart', r'pie\s*graph', r'percentage.*breakdown',
        r'proportion', r'composition', r'share.*of',
    ],
    'scatter_plot': [
        r'scatter\s*plot', r'scatter\s*chart', r'correlation',
        r'relationship\s*between',
    ],
    'histogram': [
        r'histogram', r'frequency\s*distribution',
    ],
    'comparison': [
        r'comparison\s*chart', r'compare.*groups', r'side\s*by\s*side',
        r'grouped\s*bar',
    ],
    'infographic': [
        r'infographic', r'key\s*stats', r'summary\s*card',
        r'fact\s*sheet', r'dashboard',
    ],
    'timeline': [
        r'timeline', r'chronolog', r'history.*events',
        r'milestones',
    ],
}


def detect_chart_request(query: str) -> Optional[str]:
    """Detect if the user is requesting any kind of visual (chart/graph/image) and return the type."""
    q_lower = query.lower()

    # ── First check for specific chart types (data visualization) ──
    for chart_type, patterns in CHART_PATTERNS.items():
        if any(re.search(p, q_lower) for p in patterns):
            return chart_type

    # ── Check for AI photo/image generation requests ──
    photo_patterns = [
        # Direct action verbs + visual nouns
        r'(?:generate|create|make|draw|show|build|design|render|produce|sketch|paint|illustrate)\s+(?:a\s+|an\s+|me\s+(?:a\s+|an\s+)?)?(?:image|photo|picture|illustration|figure|poster|banner|artwork|painting|portrait|wallpaper)',
        # Visual nouns + context
        r'(?:image|photo|picture|illustration|artwork|painting|portrait)\s+(?:of|for|about|showing|depicting|illustrating|representing)',
        # Polite requests
        r'(?:can you|could you|please|i want|i need|i\'d like)\s+(?:to\s+)?(?:generate|create|make|draw|show)\s+(?:a\s+|an\s+|me\s+)?',
        # "Show me" pattern
        r'show\s+me\s+(?:a\s+|an\s+)?(?:image|picture|photo)',
        # "Draw" anything (not chart-related)
        r'\b(?:draw|sketch|paint|illustrate)\s+(?:a\s+|an\s+|me\s+(?:a\s+|an\s+)?)?\w',
        # "Generate" anything visual
        r'\bgenerate\s+(?:a\s+|an\s+|me\s+(?:a\s+|an\s+)?)?(?:image|photo|picture|visual|graphic|poster|artwork)',
        # "image of" / "picture of"
        r'\b(?:image|picture|photo)\s+of\b',
    ]

    is_photo_request = any(re.search(p, q_lower) for p in photo_patterns)
    if is_photo_request:
        return 'ai_photo'

    # ── Check for general visualization requests ──
    viz_patterns = [
        r'(?:\b\w+\s+(?:image|photo|picture|illustration|artwork|painting|portrait)\b)',

        r'\bplot\s+(?:a\s+|the\s+)?',
        r'\bgraph\s+(?:of|for|showing|about)\b',
    ]
    
    is_viz_request = any(re.search(p, q_lower) for p in viz_patterns)
    if is_viz_request:
        if re.search(r'\d+', q_lower):
            return 'bar_chart'
        return 'infographic'

    return None


def parse_and_generate_chart(chart_type: str, chart_spec: Dict[str, Any]) -> Optional[str]:
    """
    Parse a chart specification and generate the chart.
    Returns image_id on success, None on failure.
    """
    # Handle AI photo generation (not a matplotlib chart)
    if chart_type == 'ai_photo':
        prompt = chart_spec.get('prompt', chart_spec.get('title', ''))
        if not prompt:
            return None
        return generate_ai_photo(prompt)
    
    if not HAS_MATPLOTLIB:
        return None

    try:
        if chart_type == 'bar_chart':
            return generate_bar_chart(
                title=chart_spec.get('title', 'Bar Chart'),
                labels=chart_spec.get('labels', []),
                values=chart_spec.get('values', []),
                ylabel=chart_spec.get('ylabel', 'Value'),
                xlabel=chart_spec.get('xlabel', ''),
                horizontal=chart_spec.get('horizontal', False)
            )
        elif chart_type == 'line_chart':
            return generate_line_chart(
                title=chart_spec.get('title', 'Line Chart'),
                x_data=chart_spec.get('x_data', []),
                y_datasets=chart_spec.get('y_datasets', []),
                xlabel=chart_spec.get('xlabel', ''),
                ylabel=chart_spec.get('ylabel', 'Value')
            )
        elif chart_type == 'pie_chart':
            return generate_pie_chart(
                title=chart_spec.get('title', 'Pie Chart'),
                labels=chart_spec.get('labels', []),
                values=chart_spec.get('values', [])
            )
        elif chart_type == 'scatter_plot':
            return generate_scatter_plot(
                title=chart_spec.get('title', 'Scatter Plot'),
                x_data=chart_spec.get('x_data', []),
                y_data=chart_spec.get('y_data', []),
                labels=chart_spec.get('labels', None),
                xlabel=chart_spec.get('xlabel', 'X'),
                ylabel=chart_spec.get('ylabel', 'Y')
            )
        elif chart_type == 'histogram':
            return generate_histogram(
                title=chart_spec.get('title', 'Histogram'),
                data=chart_spec.get('data', []),
                bins=chart_spec.get('bins', 20),
                xlabel=chart_spec.get('xlabel', 'Value'),
                ylabel=chart_spec.get('ylabel', 'Frequency')
            )
        elif chart_type == 'comparison':
            return generate_comparison_chart(
                title=chart_spec.get('title', 'Comparison'),
                categories=chart_spec.get('categories', []),
                group_names=chart_spec.get('group_names', []),
                group_values=chart_spec.get('group_values', []),
                ylabel=chart_spec.get('ylabel', 'Value')
            )
        elif chart_type == 'infographic':
            return generate_infographic(
                title=chart_spec.get('title', 'Key Facts'),
                items=chart_spec.get('items', [])
            )
        elif chart_type == 'timeline':
            return generate_timeline(
                title=chart_spec.get('title', 'Timeline'),
                events=chart_spec.get('events', [])
            )
    except Exception as e:
        print(f"[ImageGen] Error generating {chart_type}: {e}")
        return None

    return None
