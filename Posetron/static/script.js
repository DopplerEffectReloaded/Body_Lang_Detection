var menu_button = document.getElementById("menu-button")
var nav = document.getElementById("Navigation")
var menu = document.getElementById("menu")

nav.style.right = "-200px";
menu_button.onclick = function(){
    if(nav.style.right == "-200px"){
        nav.style.right = "0";
        menu.src = "../static/close.png"
    }
    else{
        nav.style.right = "-200px";
        menu.src = "../static/menu.png"
    }
}
var scroll = new SmoothScroll('a[href*="#"]', {
	speed: 1000,
	speedAsDuration: true
});