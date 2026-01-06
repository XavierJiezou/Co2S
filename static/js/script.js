document.addEventListener('DOMContentLoaded', () => {
    
    // 初始化各个功能模块
    initBibtexCopy();
    initCarousel();
    initLightbox();
    initProgressBar();

    /* ==========================================
       功能 1: BibTeX 一键复制
       ========================================== */
    function initBibtexCopy() {
        const copyBtn = document.querySelector('.copy-btn');
        const bibtexBlock = document.querySelector('bibtex');
        
        if (copyBtn && bibtexBlock) {
            copyBtn.addEventListener('click', () => {
                const textToCopy = bibtexBlock.innerText;
                navigator.clipboard.writeText(textToCopy).then(() => {
                    const originalText = copyBtn.innerText;
                    copyBtn.innerText = 'Copied!';
                    copyBtn.style.backgroundColor = '#e6fffa';
                    copyBtn.style.color = '#047857';
                    copyBtn.style.borderColor = '#047857';
                    
                    setTimeout(() => { 
                        copyBtn.innerText = originalText; 
                        copyBtn.style.backgroundColor = '';
                        copyBtn.style.color = '';
                        copyBtn.style.borderColor = '';
                    }, 2000);
                }).catch(err => {
                    console.error('Failed to copy text: ', err);
                });
            });
        }
    }

    /* ==========================================
       功能 2: 图片灯箱 (Lightbox) - 点击放大
       ========================================== */
    function initLightbox() {
        // 1. 动态创建 Lightbox 的 DOM 结构
        const lightbox = document.createElement('div');
        lightbox.id = 'lightbox-modal';
        lightbox.innerHTML = `
            <span id="lightbox-close">&times;</span>
            <img class="lightbox-content" id="lightbox-content">
        `;
        document.body.appendChild(lightbox);

        const modal = document.getElementById('lightbox-modal');
        const modalImg = document.getElementById('lightbox-content');
        const closeBtn = document.getElementById('lightbox-close');

        // 2. 为所有带有 .zoom-trigger 类的容器内的图片绑定点击事件
        // 注意：HTML中我们给img的父级div加了.zoom-trigger类，或者直接给img加了
        // 这里使用通用选择器，匹配 .zoom-trigger 下的 img，或者本身就是 .zoom-trigger 的 img
        const triggers = document.querySelectorAll('.zoom-trigger img, img.zoom-trigger');
        
        triggers.forEach(img => {
            img.style.cursor = 'zoom-in'; // 确保鼠标变手型
            img.addEventListener('click', function(e) {
                // 阻止事件冒泡，防止触发轮播图的点击
                e.stopPropagation(); 
                
                modal.style.display = "flex"; // flex 布局使图片居中
                modal.style.alignItems = "center";
                modal.style.justifyContent = "center";
                modalImg.src = this.src;
            });
        });

        // 3. 关闭逻辑
        closeBtn.onclick = function() {
            modal.style.display = "none";
        }
        
        // 点击背景也可以关闭
        modal.onclick = function(event) {
            if (event.target === modal) {
                modal.style.display = "none";
            }
        }
    }

    /* ==========================================
       功能 3: 阅读进度条 (Progress Bar)
       ========================================== */
    function initProgressBar() {
        const progressBar = document.getElementById("progress-bar");
        
        if (progressBar) {
            window.addEventListener('scroll', () => {
                const winScroll = document.body.scrollTop || document.documentElement.scrollTop;
                const height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
                const scrolled = (winScroll / height) * 100;
                progressBar.style.width = scrolled + "%";
            });
        }
    }

    /* ==========================================
       功能 4: 轮播图逻辑 (Carousel Logic)
       ========================================== */
    function initCarousel() {
        const slides = document.querySelectorAll('.carousel-slide');
        
        if (slides.length > 0) {
            let slideIndex = 1; 
            let autoPlayInterval;
            const wrapper = document.querySelector('.carousel-wrapper');
            const dots = document.querySelectorAll('.dot');
            const prevBtn = document.querySelector('.carousel-btn.prev');
            const nextBtn = document.querySelector('.carousel-btn.next');

            // 初始化显示
            showSlides(slideIndex);
            startAutoPlay();

            // 绑定按钮事件
            if(prevBtn) {
                prevBtn.addEventListener('click', () => {
                    moveSlide(-1);
                });
            }
            if(nextBtn) {
                nextBtn.addEventListener('click', () => {
                    moveSlide(1);
                });
            }

            // 绑定圆点点击事件
            dots.forEach((dot, index) => {
                dot.addEventListener('click', () => {
                    currentSlide(index + 1);
                });
            });

            // 核心切换函数
            function showSlides(n) {
                if (n > slides.length) { slideIndex = 1; }
                else if (n < 1) { slideIndex = slides.length; }
                else { slideIndex = n; }

                const translateX = -(slideIndex - 1) * 100;
                wrapper.style.transform = `translateX(${translateX}%)`;

                dots.forEach(dot => dot.classList.remove('active'));
                if (dots[slideIndex - 1]) {
                    dots[slideIndex - 1].classList.add('active');
                }
            }

            function moveSlide(n) {
                showSlides(slideIndex + n);
                resetAutoPlay();
            }

            function currentSlide(n) {
                showSlides(n);
                resetAutoPlay();
            }

            function startAutoPlay() {
                if (autoPlayInterval) clearInterval(autoPlayInterval);
                autoPlayInterval = setInterval(() => {
                    moveSlide(1);
                }, 5000); 
            }

            function resetAutoPlay() {
                clearInterval(autoPlayInterval);
                startAutoPlay();
            }
        }
    }
});

/* ==========================================
   Tab 切换逻辑 (Experimental Results)
   ========================================== */

/**
 * 切换主数据集 Tab (WHDLD, LoveDA, etc.)
 * @param {Event} evt 点击事件
 * @param {String} datasetId 对应内容div的ID
 */
function openDataset(evt, datasetId) {
    // 1. 隐藏所有数据集内容
    var contents = document.getElementsByClassName("dataset-content");
    for (var i = 0; i < contents.length; i++) {
        contents[i].classList.remove("active");
    }

    // 2. 移除所有主 Tab 按钮的 active 状态
    var buttons = document.querySelectorAll(".main-tabs .tab-btn");
    for (var i = 0; i < buttons.length; i++) {
        buttons[i].classList.remove("active");
    }

    // 3. 显示当前选中的数据集内容
    document.getElementById(datasetId).classList.add("active");

    // 4. 给当前点击的按钮添加 active 类
    evt.currentTarget.classList.add("active");
}

/**
 * 切换子 Tab (1/8, 1/4 比例等)
 * @param {Event} evt 点击事件
 * @param {String} subTabId 对应子表格div的ID
 */
function openSubTab(evt, subTabId) {
    // 1. 找到当前点击按钮所在的容器 (sub-tab-buttons)
    // 这样可以确保只影响当前数据集下的子 Tab，不影响其他数据集
    var parentContainer = evt.currentTarget.parentElement.parentElement; 
    
    // 2. 隐藏该容器下所有的子内容
    var subContents = parentContainer.getElementsByClassName("sub-tab-content");
    for (var i = 0; i < subContents.length; i++) {
        subContents[i].classList.remove("active");
    }

    // 3. 移除该容器下所有按钮的 active 状态
    var subButtons = parentContainer.querySelectorAll(".sub-tab-btn");
    for (var i = 0; i < subButtons.length; i++) {
        subButtons[i].classList.remove("active");
    }

    // 4. 显示目标子内容
    document.getElementById(subTabId).classList.add("active");

    // 5. 激活当前按钮
    evt.currentTarget.classList.add("active");
}
