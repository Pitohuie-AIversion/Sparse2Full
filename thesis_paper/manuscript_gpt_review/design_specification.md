# 附录：像素级布局统一标准与设计规范

本规范为本项目前端界面提供像素级（Pixel-Perfect）的实现标准，旨在确保不同设备、分辨率及组件间的高度一致性与可维护性。

## 1. 栅格系统 (Grid System)

采用 12 列自适应流式布局（Fluid Layout），取消固定最大宽度限制，内容宽度由容器 padding 决定。

| 断点 (Breakpoint) | 宽度范围 (Width) | 列数 (Columns) | 槽宽 (Gutter) | 容器内边距 (Padding) |
| :--- | :--- | :--- | :--- | :--- |
| **Mobile** | < 768px | 4 | 16px | 16px |
| **Tablet** | ≥ 768px | 8 | 24px | 32px |
| **Desktop** | ≥ 1024px | 12 | 24px | 48px |
| **Wide** | ≥ 1440px | 12 | 24px | 48px |

**实现规则：**
- 容器 (`.container`) 宽度始终为 100%。
- 通过增加 padding 来适应大屏幕，而非限制 max-width。
- 栅格列宽随容器宽度自动伸缩。

## 2. 间距系统 (Spacing System)

基于 **4px** 的倍数系统，严禁使用奇数像素值（如 3px, 5px）。

| 代号 (Token) | 像素值 (Pixel) | 适用场景 |
| :--- | :--- | :--- |
| `none` | 0px | 无间距 |
| `xs` | 4px | 紧凑元素间距（如图标与文字） |
| `sm` | 8px | 小组件内部间距 |
| `md` | 16px | **基础间距单位**，组件间标准距离 |
| `lg` | 24px | 区块内部间距 |
| `xl` | 32px | 大模块间距 |
| `xxl` | 48px | 章节间距 |
| `3xl` | 64px | 页面级大间距 |
| `4xl` | 80px | 英雄区域（Hero Section）间距 |
| `5xl` | 96px | 极宽松布局 |
| `6xl` | 128px | 首页最大留白 |

## 3. 排版系统 (Typography System)

### 3.1 字体族 (Font Family)
- **标题 (Heading)**: `'Space Grotesk', 'Inter', sans-serif`
- **正文 (Body)**: `'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif`
- **代码 (Code)**: `'JetBrains Mono', 'Fira Code', monospace`

### 3.2 字号层级 (Type Scale)

采用统一字号系统，不随屏幕宽度变化，确保阅读体验一致。

| 层级 (Level) | 字号 (Font-Size) | 行高 (Line-Height) | 字重 (Weight) |
| :--- | :--- | :--- | :--- |
| **H1** | 36px (4xl) | 1.25 (Tight) | Bold (700) |
| **H2** | 30px (3xl) | 1.25 (Tight) | Semibold (600) |
| **H3** | 24px (2xl) | 1.375 (Snug) | Semibold (600) |
| **H4** | 20px (xl) | 1.375 (Snug) | Medium (500) |
| **H5** | 18px (lg) | 1.5 (Normal) | Medium (500) |
| **H6** | 16px (base) | 1.5 (Normal) | Medium (500) |
| **Body** | 16px (base) | 1.5 (Normal) | Regular (400) |
| **Small** | 14px (sm) | 1.5 (Normal) | Regular (400) |
| **Caption** | 12px (xs) | 1.5 (Normal) | Regular (400) |

## 4. 组件规范 (Component Specifications)

### 4.1 按钮 (Buttons)
| 尺寸 (Size) | 高度 (Height) | 水平内边距 (Padding-X) | 字号 (Font-Size) | 圆角 (Radius) |
| :--- | :--- | :--- | :--- | :--- |
| **Small** | 32px | 12px | 14px | 4px |
| **Medium** | 40px | 16px | 16px | 4px |
| **Large** | 48px | 24px | 18px | 4px |

### 4.2 输入框 (Inputs)
- **高度**: 同按钮尺寸 (32px / 40px / 48px)
- **边框**: 1px Solid `#E2E8F0`
- **圆角**: 4px
- **内边距**: 水平 12px
- **状态**: 
  - Focus: Border `#3B82F6` + Box-shadow `0 0 0 2px rgba(59, 130, 246, 0.2)`
  - Error: Border `#EF4444`

### 4.3 卡片 (Cards)
- **内边距**: 24px (`$spacing-lg`)
- **圆角**: 8px
- **阴影**: `0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)`
- **边框**: 1px Solid `#E2E8F0` (可选，视背景色而定)

### 4.4 图标 (Icons)
- **点击区域**: 最小 44x44px (移动端)
- **视觉尺寸**: 
  - Small: 16px
  - Medium: 20px
  - Large: 24px
  - X-Large: 32px

## 5. 响应式断点 (Responsive Breakpoints)

| 断点名称 | 阈值 (min-width) | 行为描述 |
| :--- | :--- | :--- |
| **Mobile** | 0px | 默认样式，单列布局，字号紧凑 |
| **Tablet** | 768px | 栅格变为 8 列，内边距增大，字号保持 |
| **Desktop** | 1024px | 栅格变为 12 列，大屏布局，字号增大 |
| **Wide** | 1440px | 限制最大宽度，居中显示，间距最大化 |

## 6. 设计令牌 (Design Tokens)

所有设计变量已提取为 JSON 格式，位于 `design_system/design_tokens.json`。该文件是设计与开发的**唯一事实来源 (Single Source of Truth)**。

**支持格式：**
- JSON (通用)
- SCSS Variables (前端)
- CSS Custom Properties (运行时)

## 7. 前端实现 (Frontend Implementation)

样式库位于 `design_system/styles/` 目录：

- `_variables.scss`: 包含所有 Token 的 SCSS 变量与 CSS 变量。
- `_mixins.scss`: 提供 `@include media-up(tablet)` 等响应式工具。
- `_grid.scss`: 生成 `.container`, `.grid`, `.col-1` 至 `.col-12` 类。
- `_typography.scss`: 生成 `h1`-`h6`, `.text-body` 等排版类。
- `_components.scss`: 提供 `.btn`, `.input`, `.card` 等组件类。

**引用示例：**
```scss
@use "design_system/styles/variables" as *;
@use "design_system/styles/mixins" as *;

.custom-card {
  padding: $spacing-lg; // 24px
  background: $color-surface;
  border-radius: $card-radius;
  
  @include media-up(desktop) {
    padding: $spacing-xl; // 32px
  }
}
```

## 8. 验收标准 (QA Checklist)

在提交代码前，必须通过以下像素级验收：

1.  **缩放检查**: 浏览器缩放 100% 下，使用 PerfectPixel 插件覆盖设计稿，误差 < 1px。
2.  **间距检查**: 所有垂直间距必须是 4px 的倍数。
3.  **对齐检查**: 内容必须严格对齐 12 列栅格线。
4.  **字号检查**: 计算值 (Computed) 必须与规范表完全一致。
5.  **高分屏检查**: 在 Retina (2x) 屏幕下图标与边框清晰无虚边。
6.  **交互区域**: 移动端所有可点击元素触摸区域 ≥ 44x44px。

## 9. 交付物清单

完整的设计系统包包含：
1.  **设计规范文档**: 本文档 (`design_specification.md`)
2.  **样式代码库**: `design_system/styles/*.scss`
3.  **Token 数据源**: `design_system/design_tokens.json`
4.  **QA 检查表**: `design_system/qa_checklist.md`

所有资源均已适配 1×、2×、3× 屏幕密度，确保在普通屏与视网膜屏下均保持像素完美。
