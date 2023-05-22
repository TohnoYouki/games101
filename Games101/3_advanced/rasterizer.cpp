//
// Created by goksu on 4/6/19.
//

#include <algorithm>
#include "rasterizer.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>
#include <thread>

rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f> &positions)
{
    auto id = get_next_id();
    pos_buf.emplace(id, positions);

    return {id};
}

rst::ind_buf_id rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i> &indices)
{
    auto id = get_next_id();
    ind_buf.emplace(id, indices);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f> &cols)
{
    auto id = get_next_id();
    col_buf.emplace(id, cols);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_normals(const std::vector<Eigen::Vector3f>& normals)
{
    auto id = get_next_id();
    nor_buf.emplace(id, normals);

    normal_id = id;

    return {id};
}


// Bresenham's line drawing algorithm
void rst::rasterizer::draw_line(Eigen::Vector3f begin, Eigen::Vector3f end)
{
    auto x1 = begin.x();
    auto y1 = begin.y();
    auto x2 = end.x();
    auto y2 = end.y();

    Eigen::Vector3f line_color = {255, 255, 255};

    int x,y,dx,dy,dx1,dy1,px,py,xe,ye,i;

    dx=x2-x1;
    dy=y2-y1;
    dx1=fabs(dx);
    dy1=fabs(dy);
    px=2*dy1-dx1;
    py=2*dx1-dy1;

    if(dy1<=dx1)
    {
        if(dx>=0)
        {
            x=x1;
            y=y1;
            xe=x2;
        }
        else
        {
            x=x2;
            y=y2;
            xe=x1;
        }
        Eigen::Vector2i point = Eigen::Vector2i(x, y);
        set_pixel(point,line_color);
        for(i=0;x<xe;i++)
        {
            x=x+1;
            if(px<0)
            {
                px=px+2*dy1;
            }
            else
            {
                if((dx<0 && dy<0) || (dx>0 && dy>0))
                {
                    y=y+1;
                }
                else
                {
                    y=y-1;
                }
                px=px+2*(dy1-dx1);
            }
//            delay(0);
            Eigen::Vector2i point = Eigen::Vector2i(x, y);
            set_pixel(point,line_color);
        }
    }
    else
    {
        if(dy>=0)
        {
            x=x1;
            y=y1;
            ye=y2;
        }
        else
        {
            x=x2;
            y=y2;
            ye=y1;
        }
        Eigen::Vector2i point = Eigen::Vector2i(x, y);
        set_pixel(point,line_color);
        for(i=0;y<ye;i++)
        {
            y=y+1;
            if(py<=0)
            {
                py=py+2*dx1;
            }
            else
            {
                if((dx<0 && dy<0) || (dx>0 && dy>0))
                {
                    x=x+1;
                }
                else
                {
                    x=x-1;
                }
                py=py+2*(dx1-dy1);
            }
//            delay(0);
            Eigen::Vector2i point = Eigen::Vector2i(x, y);
            set_pixel(point,line_color);
        }
    }
}

auto to_vec4(const Eigen::Vector3f& v3, float w = 1.0f)
{
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}

static bool insideTriangle(int x, int y, const Vector4f* _v){
    Vector3f v[3];
    for(int i=0;i<3;i++)
        v[i] = {_v[i].x(),_v[i].y(), 1.0};
    Vector3f f0,f1,f2;
    f0 = v[1].cross(v[0]);
    f1 = v[2].cross(v[1]);
    f2 = v[0].cross(v[2]);
    Vector3f p(x,y,1.);
    if((p.dot(f0)*f0.dot(v[2])>0) && (p.dot(f1)*f1.dot(v[0])>0) && (p.dot(f2)*f2.dot(v[1])>0))
        return true;
    return false;
}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector4f* v){
    float c1 = (x*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*y + v[1].x()*v[2].y() - v[2].x()*v[1].y()) / (v[0].x()*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*v[0].y() + v[1].x()*v[2].y() - v[2].x()*v[1].y());
    float c2 = (x*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*y + v[2].x()*v[0].y() - v[0].x()*v[2].y()) / (v[1].x()*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*v[1].y() + v[2].x()*v[0].y() - v[0].x()*v[2].y());
    float c3 = (x*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*y + v[0].x()*v[1].y() - v[1].x()*v[0].y()) / (v[2].x()*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*v[2].y() + v[0].x()*v[1].y() - v[1].x()*v[0].y());
    return {c1,c2,c3};
}

void rst::rasterizer::transform_triangle(int id)
{
    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;
    Eigen::Matrix4f mvp = projection * view * model;

    for (int k = 0; k < thread_payload.orilist->size(); k++)
    {
        if (k % thread_payload.thread_num != id) { continue; }
        Triangle* t = thread_payload.orilist->operator[](k);
        Triangle* newtri = &(thread_payload.triangles[k]);

        std::array<Eigen::Vector4f, 3> mm{
                (view * model * t->v[0]),
                (view * model * t->v[1]),
                (view * model * t->v[2])
        };

        std::transform(mm.begin(), mm.end(),
            thread_payload.viewpos.begin() + 3 * k, [](auto& v) {
                return v.template head<3>();
            });

        Eigen::Vector4f v[] = {
                mvp * t->v[0],
                mvp * t->v[1],
                mvp * t->v[2]
        };
        //Homogeneous division
        for (auto& vec : v) {
            vec.x() /= vec.w();
            vec.y() /= vec.w();
            vec.z() /= vec.w();
        }

        Eigen::Matrix4f inv_trans = (view * model).inverse().transpose();
        Eigen::Vector4f n[] = {
                inv_trans * to_vec4(t->normal[0], 0.0f),
                inv_trans * to_vec4(t->normal[1], 0.0f),
                inv_trans * to_vec4(t->normal[2], 0.0f)
        };

        //Viewport transformation
        for (auto& vert : v)
        {
            vert.x() = 0.5 * width * (vert.x() + 1.0);
            vert.y() = 0.5 * height * (vert.y() + 1.0);
            vert.z() = vert.z() * f1 + f2;
        }

        for (int i = 0; i < 3; ++i)
        {
            //screen space coordinates
            newtri->setVertex(i, v[i]);
        }

        for (int i = 0; i < 3; ++i)
        {
            //view space normal
            newtri->setNormal(i, n[i].head<3>());
        }

        newtri->setColor(0, 148, 121.0, 92.0);
        newtri->setColor(1, 148, 121.0, 92.0);
        newtri->setColor(2, 148, 121.0, 92.0);

    }
}

void rst::rasterizer::draw(std::vector<Triangle *> &TriangleList) {

    thread_payload.orilist = &(TriangleList);
    thread_payload.triangles.resize(TriangleList.size());
    thread_payload.viewpos.resize(TriangleList.size() * 3);
    thread_payload.job_type = 0;
    wake_thread_job();
    thread_payload.job_type = 1;
    wake_thread_job();
}

static Eigen::Vector3f interpolate(const Vector3f & weight, const Vector3f * attribute)
{
    return attribute[0] * weight.x() + attribute[1] * weight.y() + attribute[2] * weight.z();
}

static Eigen::Vector2f interpolate(const Vector3f& weight, const Vector2f* attribute)
{
    return attribute[0] * weight.x() + attribute[1] * weight.y() + attribute[2] * weight.z();
}

//Screen space rasterization
void rst::rasterizer::rasterize_triangle(int id) 
{
    for (int k = 0; k < thread_payload.triangles.size(); k++)
    {
        const Triangle* t = &(thread_payload.triangles[k]);
        std::array<Eigen::Vector3f, 3> view_pos = { thread_payload.viewpos[k * 3],
                                                    thread_payload.viewpos[k * 3 + 1],
                                                    thread_payload.viewpos[k * 3 + 2] };
        auto v = t->toVector4();
        auto& a = v[0], & b = v[1], & c = v[2];

        if ((c.y() - a.y()) * (b.x() - a.x()) <= (c.x() - a.x()) * (b.y() - a.y())) { continue; }

        int lower_x = static_cast<int>(std::max(std::floor(std::min(std::min(a.x(), b.x()), c.x())), 0.0f));
        int lower_y = static_cast<int>(std::max(std::floor(std::min(std::min(a.y(), b.y()), c.y())), 0.0f));

        int upper_x = std::min(static_cast<int>(std::ceil(std::max(a.x(), std::max(b.x(), c.x())))), width);
        int upper_y = std::min(static_cast<int>(std::ceil(std::max(a.y(), std::max(b.y(), c.y())))), height);

        for (int j = lower_y; j < upper_y; j++) {
            if (j % thread_payload.thread_num != id) { continue; }
            for (int i = lower_x; i < upper_x; i++) {

                if (!insideTriangle(i + 0.5, j + 0.5, t->v)) { continue; }
                int index = (height - j) * width + i;

                auto [alpha, beta, gamma] = computeBarycentric2D(i + 0.5, j + 0.5, t->v);
                Vector3f weight(alpha / v[0].w(), beta / v[1].w(), gamma / v[2].w());
                weight = weight / (weight.x() + weight.y() + weight.z());
                float depth = v[0].z() * weight.x() + v[1].z() * weight.y() + v[2].z() * weight.z();

                if (depth > depth_buf[index]) { continue; }

                Eigen::Vector3f color = interpolate(weight, t->color);
                Eigen::Vector3f normal = interpolate(weight, t->normal).normalized();
                Eigen::Vector2f coords = interpolate(weight, t->tex_coords);

                fragment_shader_payload payload(color, normal, coords, texture ? &*texture : nullptr);
                payload.view_pos = interpolate(weight, view_pos.data());

                depth_buf[index] = depth;
                frame_buf[index] = fragment_shader(payload);
            }
        }
    }
}

void rst::rasterizer::wake_thread_job()
{
    std::unique_lock<std::mutex> lock(thread_payload.mutex);
    thread_payload.begin[thread_payload.switch_id] = true;
    thread_payload.begin[(thread_payload.switch_id + 1) % 2] = false;
    thread_payload.cv.notify_all();
    thread_payload.cv.wait(lock, [&] {
        return thread_payload.finished == thread_payload.thread_num; });
    thread_payload.finished = 0;
    lock.unlock();
}

void rst::rasterizer::thread_job(int id)
{
    int switch_id = 0;
    while (!thread_payload.exit) {
        std::unique_lock<std::mutex> lock(thread_payload.mutex);
        thread_payload.cv.wait(lock, [&] { 
            return thread_payload.begin[switch_id] || thread_payload.exit; });
        lock.unlock();
        if (thread_payload.exit) { break; }

        switch (thread_payload.job_type) {
        case 0:
            transform_triangle(id);
            break;
        case 1:
            rasterize_triangle(id);
            break;
        };

        lock.lock();
        thread_payload.finished += 1;
        switch_id = (switch_id + 1) % 2;
        thread_payload.switch_id = switch_id;
        thread_payload.cv.notify_all();
        lock.unlock();
    }
}

void rst::rasterizer::set_model(const Eigen::Matrix4f& m)
{
    model = m;
}

void rst::rasterizer::set_view(const Eigen::Matrix4f& v)
{
    view = v;
}

void rst::rasterizer::set_projection(const Eigen::Matrix4f& p)
{
    projection = p;
}

void rst::rasterizer::clear(rst::Buffers buff)
{
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color)
    {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth)
    {
        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());
    }
}

rst::rasterizer::rasterizer(int w, int h, int thread_num) : width(w), height(h)
{
    frame_buf.resize(w * h);
    depth_buf.resize(w * h);

    thread_payload.thread_num = thread_num;
    texture = std::nullopt;

    for (int i = 0; i < thread_num; i++) {
        render_threads.emplace_back(&rst::rasterizer::thread_job, this, i);
    }
}

int rst::rasterizer::get_index(int x, int y)
{
    return (height-y)*width + x;
}

void rst::rasterizer::set_pixel(const Vector2i &point, const Eigen::Vector3f &color)
{
    //old index: auto ind = point.y() + point.x() * width;
    int ind = (height-point.y())*width + point.x();
    frame_buf[ind] = color;
}

void rst::rasterizer::set_vertex_shader(std::function<Eigen::Vector3f(vertex_shader_payload)> vert_shader)
{
    vertex_shader = vert_shader;
}

void rst::rasterizer::set_fragment_shader(std::function<Eigen::Vector3f(fragment_shader_payload)> frag_shader)
{
    fragment_shader = frag_shader;
}

rst::rasterizer::~rasterizer() {
    thread_payload.exit = true;
    thread_payload.cv.notify_all();
    for (int i = 0; i < thread_payload.thread_num; i++) {
        render_threads[i].join();
    }
}