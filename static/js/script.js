$(document).ready(function () {
    $('#generateButton').click(function () {
        $.ajax({
            type: "POST",
            url: "/video_captionGeneration",
            dataType: "json",
            success: function (data) {
                $('#cp-textarea').text(data.caption_text);
            },
            error: function (xhr, status, error) {
                console.error(xhr.responseText);
            }
        });
    });
});
