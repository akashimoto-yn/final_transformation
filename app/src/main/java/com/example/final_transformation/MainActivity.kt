package com.example.final_transformation

import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.content.res.Resources
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.graphics.blue
import androidx.core.graphics.get
import androidx.core.graphics.green
import androidx.core.graphics.red
import kotlinx.android.synthetic.main.activity_main.*
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.calib3d.Calib3d
import org.opencv.core.*
import org.opencv.features2d.AKAZE
import org.opencv.features2d.BFMatcher
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.io.File
import java.io.FileOutputStream
import java.util.*

class MainActivity : AppCompatActivity() {

    companion object {
        const val External_Strage_REQUEST_CODE = 3
    }

    val filename = "gray_IMG_20200129_010036.png"


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)


        if (!OpenCVLoader.initDebug()) {
            //Handle initialization error
        } else {


            if (PackageManager.PERMISSION_GRANTED == ContextCompat.checkSelfPermission(applicationContext, Manifest.permission.WRITE_EXTERNAL_STORAGE)) {

            }
            else {
                ActivityCompat.requestPermissions(
                    this,
                    arrayOf(Manifest.permission.WRITE_EXTERNAL_STORAGE),
                    External_Strage_REQUEST_CODE
                )
            }



            val input1 = resources.assets.open("a.PNG_gray.PNG")
            val input2 = resources.assets.open(filename)

            var start  = System.currentTimeMillis()
            var cimg1: Bitmap = BitmapFactory.decodeStream(input1)

            //cimg1.getPixel(0,0).red
            //cimg1.getPixel(0,0).blue
            //cimg1.getPixel(0,0).green

            var cimg2: Bitmap = BitmapFactory.decodeStream(input2)

            //cimg2.getPixel(0,0).red
            //cimg2.getPixel(0,0).blue
            //cimg2.getPixel(0,0).green

            //var cimg1 = BitmapFactory.decodeResource(this.resources, R.drawable.a)
            //var cimg2 = BitmapFactory.decodeResource(this.resources, R.drawable.b)

            var matrix1 = Mat(
                cimg1.height,
                cimg1.width,
                CvType.CV_8UC3
            )  //チャンネルが3で縦横が読み込んだ画像と同じの空の行列を用意
            var matrix2 = Mat(cimg2.height, cimg2.width, CvType.CV_8UC3)

            val gray1 = Mat(cimg1.height, cimg1.width, CvType.CV_8UC1)
            val gray2 = Mat(cimg2.height, cimg2.width, CvType.CV_8UC1)

            Utils.bitmapToMat(cimg1, matrix1)  //読み込んだビットマップを行列に変換
            Utils.bitmapToMat(cimg2, matrix2)

            //Utils.matToBitmap(matrix1, cimg1)

            //Imgproc.cvtColor(matrix1, gray1, Imgproc.COLOR_RGB2GRAY)
            //Imgproc.cvtColor(matrix2, gray2, Imgproc.COLOR_RGB2GRAY)

            //Utils.matToBitmap(gray2, cimg2)
            //saveFile(createFile(), cimg2)

            //cimg1.getPixel(0,0).red
            //cimg1.getPixel(0,0).blue
            //cimg1.getPixel(0,0).green

            //Utils.matToBitmap(gray2,cimg2)

            //cimg2.getPixel(0,0).red
            //cimg2.getPixel(0,0).blue
            //cimg2.getPixel(0,0).green

            //AKAZEを利用
            val AKAZE: AKAZE = AKAZE.create()

            var keyPoint1 = MatOfKeyPoint().apply { AKAZE.detect(gray1, this) }
            var keyPoint2 = MatOfKeyPoint().apply { AKAZE.detect(gray2, this) }

            val descriptor1 = Mat().apply { AKAZE.compute(gray1, keyPoint1, this) }
            val descriptor2 = Mat().apply { AKAZE.compute(gray2, keyPoint2, this) }

            val matches = MatOfDMatch()

            BFMatcher.create(Core.NORM_HAMMING, true).match(descriptor1, descriptor2, matches)

            val sss = matches.size()

            val size: Int = matches.toArray().size - 1
            //Log.d("size", "" + size)
            val match_array = matches.toArray()

            var count: Int = 0
            var matches_list: MutableList<DMatch> = mutableListOf()
            for (i in 0..size) {
                val forward: DMatch = match_array[i]
                matches_list.add(forward)
                count++
            }

            val for_count = count - 1
            for (i in 0..for_count) {
                for (j in for_count downTo i) {
                    val dis1: DMatch = matches_list[i]
                    val dis2: DMatch = matches_list[j]

                    if (dis1.distance > dis2.distance) {
                        val a = matches_list[i]
                        //println("変換前" + dis1.distance)
                        //println("変換前" + dis2.distance)
                        matches_list[i] = matches_list[j]
                        matches_list[j] = a
                        //println("変換後" + dis1.distance)
                        //println("変換後" + dis2.distance)
                    }
                }
            }

            val asd = ((count - 1) * 0.1).toInt()
            var good: MutableList<DMatch> = mutableListOf()
            for (i in 0..asd) {
                good.add(matches_list[i])
            }

            //println(good)

            var h: Mat
            var keylist1: MutableList<KeyPoint> = mutableListOf()
            var pointlist1: MutableList<Point> = mutableListOf()


            var outkey1 = keyPoint1.toArray()

            for (i in outkey1) {
                keylist1.add(i)
            }

            for (m in good) {
                pointlist1.add(keylist1[m.queryIdx].pt)
            }

            var src_pts = MatOfPoint2f()
            src_pts.fromList(pointlist1)

            var outkey2 = keyPoint2.toArray()
            var keylist2: MutableList<KeyPoint> = mutableListOf()
            var pointlist2: MutableList<Point> = mutableListOf()

            for (i in outkey2) {
                keylist2.add(i)
            }

            for (m in good) {
                pointlist2.add(keylist2[m.trainIdx].pt)
            }


            var dst_pts = MatOfPoint2f()
            dst_pts.fromList(pointlist2)


            h = Calib3d.findHomography(dst_pts, src_pts, Calib3d.RANSAC, 1.0)

            var dst_img = Mat()

            val dsize = Size(cimg1.width.toDouble(), cimg1.height.toDouble())

            Imgproc.warpPerspective(matrix2, dst_img, h, dsize)  //(matrix1.size())

            Utils.matToBitmap(dst_img, cimg1)

            //計測したい処理を記述

            var end = System.currentTimeMillis()
            println((end - start).toString() + "ms")

            samimg.setImageBitmap(cimg1)

            saveFile(createFile(), cimg1)

        }
    }

    private fun createFile(): File {
        val dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DCIM)
        return File(dir, "re_re_gray.png")
    }

    private fun saveFile(f: File, bitmap: Bitmap) {

        var bit: Bitmap = bitmap

        bit.height
        bit.width


        val ops = FileOutputStream(f)

        bit.compress(Bitmap.CompressFormat.PNG, 100, ops)


        ops.close()

        //ギャラリーからもアクセスできるように、画像データとしてAndroidに登録
        val contextValues = ContentValues().apply {
            put(MediaStore.Images.Media.MIME_TYPE, "image/png")
            put("_data", f.absolutePath)
        }

        contentResolver.insert(
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contextValues
        )

    }



}
